import unittest
from unittest.mock import patch

import DonutPromptReceiver


class PromptReceiverIsolationTests(unittest.TestCase):
    def test_import_does_not_start_an_http_server(self):
        self.assertIsNone(DonutPromptReceiver.PromptReceiverServer._server)

    def test_claude_bridge_allows_loopback_only(self):
        self.assertEqual(
            DonutPromptReceiver._local_outgoing_url("http", "localhost", 5728),
            "http://localhost:5728/prompt",
        )
        self.assertEqual(
            DonutPromptReceiver._local_outgoing_url("https", "127.0.0.1", 5728),
            "https://127.0.0.1:5728/prompt",
        )
        self.assertEqual(
            DonutPromptReceiver._local_outgoing_url("http", "::1", 5728),
            "http://[::1]:5728/prompt",
        )
        for host in ("example.com", "192.168.1.10", "8.8.8.8"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    DonutPromptReceiver._local_outgoing_url("https", host, 443)

    @patch("DonutPromptReceiver.requests.request")
    def test_comfyui_requests_target_only_the_fixed_loopback_api(self, request):
        DonutPromptReceiver._comfyui_request("POST", "/queue", timeout=10, json_body={"clear": True})

        request.assert_called_once_with(
            "POST",
            "http://127.0.0.1:8188/queue",
            timeout=10,
            json={"clear": True},
        )


if __name__ == "__main__":
    unittest.main()
