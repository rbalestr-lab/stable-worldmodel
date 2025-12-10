"""Pytest configuration to mock torchcodec for environments that don't need it."""
import sys

# Create mock torchcodec module before any tests run
class MockVideoDecoder:
    def __init__(self, *args, **kwargs):
        pass

class MockAudioDecoder:
    def __init__(self, *args, **kwargs):
        pass

class MockDecoders:
    VideoDecoder = MockVideoDecoder
    AudioDecoder = MockAudioDecoder

class MockTorchcodec:
    decoders = MockDecoders()

# Install the mock in sys.modules
if 'torchcodec' not in sys.modules:
    sys.modules['torchcodec'] = MockTorchcodec()
    sys.modules['torchcodec.decoders'] = MockDecoders()
