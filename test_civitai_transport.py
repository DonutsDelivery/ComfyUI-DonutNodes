import unittest
from unittest.mock import patch

from shared import civitai_transport


class FakeResponse:
    def __init__(self, *, redirect=False, location=None):
        self.is_redirect = redirect
        self.headers = {} if location is None else {"Location": location}
        self.closed = False

    def close(self):
        self.closed = True


class CivitAITransportTests(unittest.TestCase):
    def test_accepts_https_urls_on_civitai_hosts(self):
        for url in (
            "https://civitai.com/api/v1/models",
            "https://image.civitai.com/x.webp",
            "https://b2.civitai.com/file/model.safetensors",
            "https://civitai.red/api/v1/models",
        ):
            self.assertEqual(civitai_transport.validate_civitai_url(url), url)

    def test_rejects_non_https_untrusted_and_malformed_urls(self):
        for url in (
            "http://civitai.com/api/v1/models",
            "file:///etc/passwd",
            "https://civitai.com.evil.example/model",
            "https://civitai.com:444/model",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    civitai_transport.validate_civitai_url(url)

    @patch("shared.civitai_transport.requests.request")
    def test_validates_each_redirect_hop(self, request):
        redirect = FakeResponse(
            redirect=True,
            location="https://image.civitai.com/file/model.safetensors",
        )
        terminal = FakeResponse()
        request.side_effect = [redirect, terminal]

        response = civitai_transport.civitai_request(
            "GET", "https://civitai.com/api/download/models/1"
        )

        self.assertIs(response, terminal)
        self.assertTrue(redirect.closed)
        self.assertEqual(request.call_count, 2)
        self.assertFalse(request.call_args.kwargs["allow_redirects"])

    @patch("shared.civitai_transport.requests.request")
    def test_rejects_untrusted_redirect_before_following_it(self, request):
        redirect = FakeResponse(redirect=True, location="https://evil.example/model")
        request.return_value = redirect

        with self.assertRaises(ValueError):
            civitai_transport.civitai_request(
                "GET", "https://civitai.com/api/download/models/1"
            )

        self.assertTrue(redirect.closed)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
