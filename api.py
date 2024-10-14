import os
import re
import logging
import base64
import torch
import torchaudio
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from io import BytesIO
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from transcription_helpers import transcribe_batched

app = Flask(__name__)
api = Api(app)

mtypes = {"cpu": "int8", "cuda": "int8"}

class Transcribe(Resource):
    def post(self):
        data = request.get_json()
        if 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400

        audio_data = base64.b64decode(data['audio'])
        no_stem = data.get('no_stem', True)
        suppress_numerals = data.get('suppress_numerals', False)
        model_name = data.get('whisper_model', 'medium.en')
        batch_size = data.get('batch_size', 8)
        language = data.get('language', None)
        device = data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        language = process_language_arg(language, model_name)

        audio_path = os.path.join('uploads', 'audio.wav')
        with open(audio_path, 'wb') as f:
            f.write(audio_data)

        if no_stem:
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
            )

            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
                )
                vocal_target = audio_path
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio_path))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio_path

        whisper_results, language, audio_waveform = transcribe_batched(
            vocal_target,
            language,
            batch_size,
            model_name,
            mtypes[device],
            suppress_numerals,
            device,
        )

        alignment_model, alignment_tokenizer = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        audio_waveform = (
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device)
        )
        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=batch_size
        )

        del alignment_model
        torch.cuda.empty_cache()

        full_transcript = "".join(segment["text"] for segment in whisper_results)

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs")
        os.makedirs(temp_path, exist_ok=True)
        torchaudio.save(
            os.path.join(temp_path, "mono_file.wav"),
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            logging.warning(
                f"Punctuation restoration is not available for {language} language. Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        transcript_content = []
        srt_content = []

        transcript_buffer = BytesIO()
        get_speaker_aware_transcript(ssm, transcript_buffer)
        transcript_content = transcript_buffer.getvalue().decode('utf-8-sig')

        srt_buffer = BytesIO()
        write_srt(ssm, srt_buffer)
        srt_content = srt_buffer.getvalue().decode('utf-8-sig')

        return jsonify({
            "transcript": transcript_content,
            "srt": srt_content
        })

api.add_resource(Transcribe, '/transcribe')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


#   curl -X POST -H "Content-Type: application/json" -d '{
#       "audio": "'$(base64 -w 0 path_to_audio_file.wav)'",
#       "no_stem": true,
#       "suppress_numerals": false,
#       "whisper_model": "medium.en",
#       "batch_size": 8,
#       "language": "en",
#       "device": "cuda"
#   }' http://localhost:5000/transcribe
