"""Offline regression checks for the published protocol and recovery boundary."""
import json
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gemma_postformat import recover_route
import snips_eval as snips
import replicate_snips as portable
from analyze_snips import classification


def response(text, finish='STOP'):
    return {'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': text}]}}]}


class SnipsTests(unittest.TestCase):
    def test_fences_and_valid_answers(self):
        value = '{"route":"PlayMusic"}'
        for text in (value, '```json\n'+value+'\n```', value+'\n```'):
            self.assertEqual(recover_route(response(text), snips.CRITERIA), 'PlayMusic')

    def test_recovery_rejects_semantic_changes(self):
        for text in ('{"route":"unknown"}', '{"route":"PlayMusic","extra":1}',
                     '{"route":"PlayMusic","route":"GetWeather"}',
                     'Answer: {"route":"PlayMusic"}', '{"route":null}', '{"route":'):
            self.assertIsNone(recover_route(response(text), snips.CRITERIA))
        self.assertIsNone(recover_route(response('{"route":"PlayMusic"}', 'MAX_TOKENS'), snips.CRITERIA))

    def test_invalid_is_in_denominator(self):
        m = classification(['PlayMusic', 'PlayMusic'], ['PlayMusic', None], snips.CRITERIA)
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['confusion']['PlayMusic']['INVALID'], 1)

    def test_dispatch_stops_on_transport_failure_and_cannot_repeat(self):
        case = {'id': 'test-1', 'text': 'Music please', 'expected': 'PlayMusic'}
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            dest = bundle / 'deberta'
            dest.mkdir()
            request = [{'id': case['id'], 'payload': portable.payload('deberta', case)}]
            (dest / 'requests.json').write_text(json.dumps(request))
            meta = {'request_sha256': {'deberta': portable.prior.digest(dest / 'requests.json')}}
            with patch.object(portable, 'verify', return_value=([case], meta)), \
                 patch.object(portable.urllib.request.OpenerDirector, 'open', side_effect=TimeoutError):
                with self.assertRaises(SystemExit):
                    portable.run(bundle, 'deberta')
                rows = json.loads((dest / 'results.json').read_text())
                self.assertEqual(rows[0]['error'], 'TimeoutError')
                self.assertEqual(len((dest / 'attempts.jsonl').read_text().splitlines()), 1)
                with self.assertRaises(FileExistsError):
                    portable.run(bundle, 'deberta')

    def test_portable_uses_frozen_payload_and_parser(self):
        self.assertIs(portable.payload, snips.payload)
        self.assertIs(portable.normalize, snips.normalize)
        case = {'text': 'Play some jazz'}
        for model in ('gemini', 'gemma'):
            config = portable.payload(model, case)['generationConfig']
            self.assertEqual(config['responseJsonSchema']['properties']['route']['enum'], list(snips.CRITERIA))
            self.assertEqual(config['temperature'], 1)
            self.assertEqual(config['thinkingConfig']['thinkingLevel'], 'HIGH' if model == 'gemma' else 'MINIMAL')
        self.assertEqual(portable.payload('jev', case)['questions']['route']['criteria'], snips.CRITERIA)


if __name__ == '__main__':
    unittest.main()
