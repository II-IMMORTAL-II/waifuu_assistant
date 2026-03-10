import unittest
from pathlib import Path
import shutil
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat_service import ChatService
from config import Config
from app.models import ChatMessage

# Mock services
class MockGroqService:
    def __init__(self, api_key):
        pass
    def generate_response(self, prompt):
        return "Mock response"

class MockRealtimeGroqService:
    def __init__(self):
        pass

class TestChatService(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for chats
        self.test_chats_dir = Path("test_chats_data")
        Config.CHATS_DATA_DIR = self.test_chats_dir
        
        # Patch services in the module we are testing
        # Since we import ChatService, it imports GroqService. 
        # We need to monkeypatch the instance or class used by ChatService.
        # However, ChatService instantiates them in __init__.
        # We can patch the classes in the module before instantiating.
        
        import app.services.chat_service as chat_service_module
        chat_service_module.GroqService = MockGroqService
        chat_service_module.RealtimeGroqService = MockRealtimeGroqService
        
        self.chat_service = ChatService()

    def tearDown(self):
        # Clean up
        if self.test_chats_dir.exists():
            shutil.rmtree(self.test_chats_dir)

    def test_session_creation(self):
        session_id = "test_session_1"
        history = self.chat_service.get_or_create_session(session_id)
        self.assertIsNotNone(history)
        self.assertEqual(history.session_id, session_id)
        self.assertEqual(len(history.messages), 0)

    def test_add_message(self):
        session_id = "test_session_2"
        self.chat_service.add_message(session_id, "user", "Hello")
        
        history = self.chat_service.get_chat_history(session_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello")
        
        # Verify persistence
        session_file = self.test_chats_dir / f"{session_id}.json"
        self.assertTrue(session_file.exists())

    def test_process_message(self):
        session_id = "test_session_3"
        response = self.chat_service.process_message(session_id, "Hello AI")
        
        self.assertEqual(response, "Mock response")
        history = self.chat_service.get_chat_history(session_id)
        self.assertEqual(len(history), 2) # User + Assistant
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "Mock response")

if __name__ == '__main__':
    unittest.main()
