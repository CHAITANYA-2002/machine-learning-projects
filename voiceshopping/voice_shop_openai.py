import speech_recognition as sr
import pyttsx3
import json
from typing import List, Dict, Optional
import os
from openai import OpenAI

class VoiceShoppingAssistantAI:
    def __init__(self, openai_api_key: str):
        print("🤖 Initializing Voice Shopping Assistant...")
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        print("✓ OpenAI connected")
        
        # Initialize speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        print("✓ Voice systems ready")
        
        # Shopping cart and conversation history
        self.cart = []
        self.conversation_history = []
        self.last_search_results = []
        
        # System prompt for OpenAI
        self.system_prompt = """You are a helpful voice shopping assistant for Australian supermarkets (Coles, Woolworths, and ALDI). 

Your job is to:
1. Help users find grocery items they want to buy
2. Extract item names and quantities from their requests
3. Present product options clearly
4. Manage their shopping cart
5. Guide them through checkout

When the user mentions an item, respond with a JSON object:
{
    "action": "search|add_to_cart|show_cart|checkout|clarify|general",
    "item": "item name",
    "quantity": number,
    "response": "what to say to the user",
    "needs_clarification": true/false
}

Be conversational, friendly, and efficient. Always confirm actions before executing them."""
        
        # Initialize conversation
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })
        print("✓ Assistant ready!\n")
    
    def speak(self, text: str):
        """Convert text to speech"""
        print(f"🤖 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self) -> str:
        """Listen to user voice input"""
        with sr.Microphone() as source:
            print("\n🎤 Listening... (speak now)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio)
                print(f"👤 You: {text}")
                return text
            except sr.WaitTimeoutError:
                print("⏱️  Timeout - no speech detected")
                return ""
            except sr.UnknownValueError:
                self.speak("Sorry, I couldn't understand that. Could you repeat?")
                return ""
            except Exception as e:
                print(f"❌ Error: {e}")
                return ""
    
    def get_ai_response(self, user_message: str) -> Dict:
        """Get intelligent response from OpenAI"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Add context about current cart
        context = f"\n\nCurrent cart: {json.dumps(self.cart, indent=2)}"
        if self.last_search_results:
            context += f"\n\nLast search results: {json.dumps(self.last_search_results, indent=2)}"
        
        # Get response from OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model
                messages=self.conversation_history + [{
                    "role": "system",
                    "content": context
                }],
                temperature=0.7,
                max_tokens=300
            )
            
            ai_message = response.choices[0].message.content
            
            # Add AI response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_message
            })
            
            # Try to parse as JSON, fallback to text response
            try:
                return json.loads(ai_message)
            except json.JSONDecodeError:
                # If not JSON, create a general response
                return {
                    "action": "general",
                    "response": ai_message,
                    "needs_clarification": False
                }
        
        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            return {
                "action": "error",
                "response": "I'm having trouble processing that. Could you rephrase?",
                "needs_clarification": True
            }
    
    def search_products(self, query: str) -> List[Dict]:
        """
        Search for products across supermarkets
        NOTE: Replace with actual web scraping implementation
        """
        print(f"🔍 Searching for: {query}")
        
        # Mock results - replace with real scraping
        mock_results = [
            {
                "id": f"{query}_coles",
                "name": f"{query.title()}",
                "store": "Coles",
                "price": 3.50,
                "url": "https://coles.com.au/...",
                "in_stock": True,
                "size": "500g"
            },
            {
                "id": f"{query}_woolworths",
                "name": f"{query.title()} - Premium",
                "store": "Woolworths",
                "price": 3.80,
                "url": "https://woolworths.com.au/...",
                "in_stock": True,
                "size": "500g"
            },
            {
                "id": f"{query}_aldi",
                "name": f"{query.title()} - Value Pack",
                "store": "ALDI",
                "price": 2.99,
                "url": "https://aldi.com.au/...",
                "in_stock": True,
                "size": "600g"
            }
        ]
        
        self.last_search_results = mock_results
        return mock_results
    
    def present_options_ai(self, products: List[Dict], item_name: str, quantity: int) -> Optional[Dict]:
        """Present options and let AI help with selection"""
        if not products:
            self.speak(f"Sorry, I couldn't find {item_name} at any store.")
            return None
        
        # Create summary for AI
        options_summary = "\n".join([
            f"Option {i+1}: {p['name']} at {p['store']} - ${p['price']} ({p['size']})"
            for i, p in enumerate(products)
        ])
        
        self.speak(f"I found {len(products)} options for {item_name}:")
        for i, product in enumerate(products, 1):
            self.speak(f"Option {i}: {product['name']} at {product['store']} for ${product['price']}")
        
        self.speak("Which one would you like, or should I pick the cheapest?")
        
        # Get user choice
        user_choice = self.listen()
        if not user_choice:
            return None
        
        # Use AI to interpret choice
        ai_response = self.get_ai_response(
            f"User wants to select from these options:\n{options_summary}\n\nThey said: '{user_choice}'\n\nWhich product should be selected? Respond with the option number (1-{len(products)}) or 'cheapest'."
        )
        
        response_text = ai_response.get("response", "").lower()
        
        # Parse selection
        if "cheap" in response_text or "lowest" in response_text:
            selected = min(products, key=lambda x: x['price'])
        else:
            # Try to find number in response
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                idx = int(numbers[0]) - 1
                selected = products[idx] if 0 <= idx < len(products) else products[0]
            else:
                selected = products[0]
        
        selected['quantity'] = quantity
        self.speak(f"Great! Selected {selected['name']} from {selected['store']}.")
        return selected
    
    def add_to_cart(self, item: Dict):
        """Add item to shopping cart"""
        self.cart.append(item)
        total_price = item['price'] * item['quantity']
        self.speak(f"Added {item['quantity']} {item['name']} to your cart. That's ${total_price:.2f}")
    
    def show_cart(self):
        """Display current cart"""
        if not self.cart:
            self.speak("Your cart is empty.")
            return
        
        self.speak("Here's what's in your cart:")
        total = 0
        
        # Group by store
        by_store = {}
        for item in self.cart:
            store = item['store']
            if store not in by_store:
                by_store[store] = []
            by_store[store].append(item)
        
        for store, items in by_store.items():
            self.speak(f"From {store}:")
            store_total = 0
            for item in items:
                item_total = item['price'] * item['quantity']
                store_total += item_total
                self.speak(f"  {item['quantity']} {item['name']}: ${item_total:.2f}")
            total += store_total
        
        self.speak(f"Your total is ${total:.2f}")
    
    def checkout(self):
        """Process checkout"""
        if not self.cart:
            self.speak("Your cart is empty. Add some items first!")
            return False
        
        self.show_cart()
        self.speak("Would you like to proceed with checkout?")
        
        response = self.listen()
        
        # Use AI to understand confirmation
        ai_response = self.get_ai_response(
            f"User said: '{response}'. Are they confirming checkout? Respond with 'yes' or 'no'."
        )
        
        if "yes" in ai_response.get("response", "").lower():
            self.speak("Processing your order across multiple stores. In a real implementation, this would place orders at each supermarket.")
            
            # Group by store and show order summary
            by_store = {}
            for item in self.cart:
                store = item['store']
                if store not in by_store:
                    by_store[store] = []
                by_store[store].append(item)
            
            for store, items in by_store.items():
                self.speak(f"Order placed at {store} for {len(items)} items.")
            
            self.speak("All orders placed successfully! Thank you for shopping.")
            self.cart = []
            return True
        else:
            self.speak("Okay, your items are still in your cart. Anything else you'd like to add?")
            return False
    
    def run(self):
        """Main conversation loop with AI"""
        self.speak("Hello! I'm your AI-powered shopping assistant for Coles, Woolworths, and ALDI.")
        self.speak("What would you like to order today?")
        
        while True:
            user_input = self.listen()
            
            if not user_input:
                continue
            
            # Get AI interpretation
            ai_response = self.get_ai_response(user_input)
            action = ai_response.get("action", "general")
            
            # Check for exit
            if action == "exit" or any(word in user_input.lower() for word in ["exit", "quit", "goodbye", "bye"]):
                self.speak("Thank you for using the shopping assistant. Goodbye!")
                break
            
            # Handle different actions
            if action == "search":
                item = ai_response.get("item", "")
                quantity = ai_response.get("quantity", 1)
                
                self.speak(ai_response.get("response", f"Searching for {item}..."))
                products = self.search_products(item)
                
                selected_item = self.present_options_ai(products, item, quantity)
                
                if selected_item:
                    self.speak("Would you like to add this to your cart?")
                    confirm = self.listen()
                    
                    confirm_response = self.get_ai_response(
                        f"User said: '{confirm}'. Are they confirming to add to cart? Respond 'yes' or 'no'."
                    )
                    
                    if "yes" in confirm_response.get("response", "").lower():
                        self.add_to_cart(selected_item)
            
            elif action == "show_cart":
                self.show_cart()
                self.speak(ai_response.get("response", "What else would you like to do?"))
            
            elif action == "checkout":
                self.checkout()
            
            elif action == "add_to_cart":
                # Direct add to cart from last search
                if self.last_search_results:
                    self.speak(ai_response.get("response", "Adding to cart..."))
                else:
                    self.speak("Please search for an item first.")
            
            else:
                # General conversation
                self.speak(ai_response.get("response", "I'm not sure what you'd like to do. Could you clarify?"))


# ==========================================
# MAIN PROGRAM START
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("  VOICE SHOPPING ASSISTANT - Australian Supermarkets")
    print("="*60)
    
    # Get API key - REPLACE WITH YOUR KEY
    # IMPORTANT: Never hard-code API keys in source code. Prefer using
    # environment variables (e.g., OPENAI_API_KEY) or a secrets manager.
    # This placeholder will cause the script to ask you to set the
    # OPENAI_API_KEY environment variable or replace the placeholder.
    api_key = os.getenv("OPENAI_API_KEY") or "PUT-YOUR-API-KEY-HERE"
    
    # Check if API key is set
    if api_key == "PUT-YOUR-API-KEY-HERE":
        print("\n❌ ERROR: Please set your OpenAI API key!")
        print("\n📝 How to fix:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Replace 'PUT-YOUR-API-KEY-HERE' in the code with your actual key")
        print("   OR set environment variable: set OPENAI_API_KEY=your-key-here")
        print("\n")
        input("Press Enter to exit...")
        exit(1)
    
    try:
        # Create and run the assistant
        assistant = VoiceShoppingAssistantAI(api_key)
        assistant.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program stopped by user (Ctrl+C)")
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("\n💡 Common issues:")
        print("- Microphone not connected or permissions denied")
        print("- OpenAI API key invalid or no credits")
        print("- Internet connection required for voice recognition and AI")
        input("\nPress Enter to exit...")