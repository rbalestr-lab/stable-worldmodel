"""Pytest configuration for stable-worldmodel tests."""
import sys
from types import ModuleType
from importlib.machinery import ModuleSpec

# Mock torchcodec to avoid import errors in environments that don't need video encoding
class MockVideoDecoder:
    def __init__(self, *args, **kwargs):
        pass

class MockAudioDecoder:
    def __init__(self, *args, **kwargs):
        pass

# Create mock modules with proper __spec__ attributes
if 'torchcodec' not in sys.modules:
    # Create mock torchcodec module
    mock_torchcodec = ModuleType('torchcodec')
    mock_torchcodec.__spec__ = ModuleSpec('torchcodec', None)

    # Create mock decoders submodule
    mock_decoders = ModuleType('torchcodec.decoders')
    mock_decoders.__spec__ = ModuleSpec('torchcodec.decoders', None)
    mock_decoders.VideoDecoder = MockVideoDecoder
    mock_decoders.AudioDecoder = MockAudioDecoder

    # Set decoders as attribute of torchcodec
    mock_torchcodec.decoders = mock_decoders

    # Install in sys.modules
    sys.modules['torchcodec'] = mock_torchcodec
    sys.modules['torchcodec.decoders'] = mock_decoders
