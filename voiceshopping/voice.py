"""voiceshopping.voice

A small voice-driven shopping assistant prototype.
This module demonstrates a conversational assistant that listens to the
user, searches for mock products, presents options, adds items to a
shopping cart, and simulates checkout.

Notes:
- The product search is a placeholder returning mock data; integrate
  real supermarket APIs or web scraping for production use.
- This file focuses on clarity and is heavily commented for learning.
"""

import speech_recognition as sr
import pyttsx3
import json
from typing import List, Dict, Tuple
import re


class VoiceShoppingAssistant:
    """Voice-driven shopping assistant.

    Attributes
    ----------
    recognizer: sr.Recognizer
        The SpeechRecognition recognizer instance used to capture audio.
    tts_engine: pyttsx3.Engine
        Text-to-speech engine used to speak back to the user.
    cart: List[Dict]
        In-memory list representing the user's shopping cart.
    conversation_state: str
        Optional state variable to track conversation phase (e.g., greeting).
    supermarkets: List[str]
        A prioritized list of supermarkets to query (placeholder).
    """

    def __init__(self):
        # Initialize speech recognition (microphone input) and TTS engine
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        # Set speaking rate: lower numbers -> slower speech
        self.tts_engine.setProperty('rate', 150)

        # Simple in-memory cart (each item is a dict with keys like name, price, store, quantity)
        self.cart: List[Dict] = []

        # Conversation phase state (not heavily used here, but handy for expansion)
        self.conversation_state = "greeting"

        # Supermarket priority/order - can be used by search logic
        self.supermarkets = ["coles", "woolworths", "aldi"]

    def speak(self, text: str) -> None:
        """Convert text to speech and print to console.

        Parameters
        ----------
        text : str
            The message to speak aloud and print for debugging.
        """
        # Print to console for visibility during development/testing
        print(f"Assistant: {text}")
        # Use pyttsx3 to perform TTS synchronously
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self) -> str:
        """Listen for user voice input and transcribe to lowercase string.

        Returns
        -------
        str
            Transcribed text in lowercase, or empty string on failure.

        Implementation notes
        --------------------
        - Uses a short ambient noise adjustment to help the recognizer adapt
          to background noise.
        - Uses a timeout to avoid blocking forever if no speech is detected.
        - Handles common exceptions and returns an empty string on failure.
        """
        # Use the system's default microphone as the audio source
        with sr.Microphone() as source:
            print("Listening...")
            # Adjust for ambient noise for a short duration to improve accuracy
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            try:
                # Listen for a phrase: raise if nothing is said within timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                # Use Google Web Speech API for transcription - requires internet
                text = self.recognizer.recognize_google(audio)
                print(f"User: {text}")
                return text.lower()

            except sr.WaitTimeoutError:
                # No speech was detected within the timeout window
                self.speak("I didn't hear anything. Please try again.")
                return ""

            except sr.UnknownValueError:
                # Speech was unintelligible to the recognizer
                self.speak("Sorry, I couldn't understand that.")
                return ""

            except Exception as e:
                # Catch-all for other issues (network errors, API errors, etc.)
                # Avoid raising to keep the conversation loop resilient during demo
                print(f"Error: {e}")
                return ""

    def search_products(self, query: str) -> List[Dict]:
        """Search for products across supermarkets.

        This is a placeholder stub that returns mock products for the given
        query. Replace this with real API calls or web scraping to get live
        data from supermarket websites.

        Parameters
        ----------
        query : str
            The product name or search phrase provided by the user.

        Returns
        -------
        List[Dict]
            A list of product dictionaries containing at least name, store,
            price, url, and in_stock keys.
        """
        # Placeholder results - structured to look like real scraped/API data
        mock_results: List[Dict] = [
            {
                "name": f"{query.title()}",
                "store": "Coles",
                "price": 3.50,
                "url": "https://coles.com.au/...",
                "in_stock": True,
            },
            {
                "name": f"{query.title()} - Premium",
                "store": "Woolworths",
                "price": 3.80,
                "url": "https://woolworths.com.au/...",
                "in_stock": True,
            },
            {
                "name": f"{query.title()} - Value Pack",
                "store": "ALDI",
                "price": 2.99,
                "url": "https://aldi.com.au/...",
                "in_stock": True,
            },
        ]

        return mock_results

    def extract_item_and_quantity(self, text: str) -> Tuple[str, int]:
        """Extract a product name and optional quantity from free-form text.

        Examples the method handles:
        - "2 cans of tuna" -> ("cans of tuna", 2)
        - "one dozen eggs" -> ("one dozen eggs", 1) # quantity extraction only handles digits
        - "apples" -> ("apples", 1)

        Notes
        -----
        - The simple regex looks for a digit followed by the rest of the string.
          This is intentionally simple; expand to support words-to-numbers if needed.

        Parameters
        ----------
        text : str
            Raw user input potentially containing quantity and item.

        Returns
        -------
        (item: str, quantity: int)
            Tuple with the cleaned item name and an integer quantity (default 1).
        """
        # Match patterns like '2 apples' or '3 of milk' capturing the number and the remainder
        quantity_match = re.search(r"(\d+)\s+(?:of\s+)?(.+)", text)

        if quantity_match:
            # Group 1 is the quantity (digits); Group 2 is the item text
            quantity = int(quantity_match.group(1))
            item = quantity_match.group(2).strip()
        else:
            # Default quantity is 1 when no explicit number is found
            quantity = 1
            item = text.strip()

        # Remove polite filler words to keep the item name concise
        item = re.sub(r"\b(please|can i have|i want|i need|get me)\b", "", item, flags=re.IGNORECASE).strip()

        return item, quantity

    def present_options(self, products: List[Dict], item_name: str, quantity: int) -> Dict:
        """Present product choices to the user and return the selected one.

        Interaction flow:
        1. Read out the available options with store and price.
        2. Ask the user to pick by number or say 'cheapest'.
        3. Parse the user's response and pick the matching product.

        Parameters
        ----------
        products : List[Dict]
            List of candidate products returned by `search_products`.
        item_name : str
            The human-friendly name of the item being searched for.
        quantity : int
            The quantity the user requested.

        Returns
        -------
        Dict
            The selected product dict (with 'quantity' added) or None if no products.
        """
        if not products:
            self.speak(f"Sorry, I couldn't find {item_name} at any store.")
            return None

        # Announce how many matches were found and list them
        self.speak(f"I found {item_name} at {len(products)} stores. Here are your options:")

        for i, product in enumerate(products, 1):
            # Read out each option with index to allow number-based selection
            self.speak(f"Option {i}: {product['name']} at {product['store']} for ${product['price']}")

        self.speak("Which option would you like? Say the number, or say 'cheapest' for the lowest price.")

        choice = self.listen()

        # If user says 'cheapest' or similar, pick the lowest priced item
        if "cheap" in choice or "lowest" in choice:
            selected = min(products, key=lambda x: x['price'])
            self.speak(f"Great! I'll select the cheapest option from {selected['store']}.")
        else:
            # Try to extract a number from the spoken choice (e.g., 'option 2')
            number_match = re.search(r"\d+", choice)
            if number_match:
                idx = int(number_match.group()) - 1
                if 0 <= idx < len(products):
                    selected = products[idx]
                else:
                    # Out-of-range number; fall back to first option and inform the user
                    self.speak("Invalid option. Selecting the first one.")
                    selected = products[0]
            else:
                # No selection found: default to the first product
                selected = products[0]

        # Attach the requested quantity so downstream code knows how many to add
        selected['quantity'] = quantity
        return selected

    def add_to_cart(self, item: Dict) -> None:
        """Add an item dictionary to the in-memory cart and announce it.

        Parameters
        ----------
        item : Dict
            Product dictionary expected to contain at least 'name', 'price',
            'store' and 'quantity' keys.
        """
        self.cart.append(item)
        total_price = item['price'] * item['quantity']
        self.speak(f"Added {item['quantity']} {item['name']} from {item['store']} to your cart. That's ${total_price:.2f}")

    def show_cart(self) -> None:
        """Speak the current cart contents and the total price."""
        if not self.cart:
            self.speak("Your cart is empty.")
            return

        self.speak("Here's what's in your cart:")
        total = 0.0
        for item in self.cart:
            # Compute subtotal per item and accumulate
            item_total = item['price'] * item['quantity']
            total += item_total
            self.speak(f"{item['quantity']} {item['name']} from {item['store']}: ${item_total:.2f}")

        # Announce the cart total rounded to two decimals
        self.speak(f"Your total is ${total:.2f}")

    def checkout(self) -> bool:
        """Simulate a checkout interaction.

        This method shows the cart and asks the user to confirm. In a real
        implementation this is where integration with the supermarket's
        ordering API or a web automation tool would be performed.

        Returns
        -------
        bool
            True if the user confirmed checkout and order was "placed", False otherwise.
        """
        self.show_cart()
        self.speak("Would you like to proceed with checkout? Say yes or no.")

        response = self.listen()

        # Simple set of affirmative responses handled here
        if "yes" in response or "yeah" in response or "sure" in response:
            self.speak("Processing your order. This would normally integrate with the supermarket websites.")
            # Placeholder: in production, place the order and check responses
            self.speak("Order placed successfully! Thank you for shopping.")
            # Clear cart after successful order
            self.cart = []
            return True
        else:
            self.speak("Okay, your items are still in your cart.")
            return False

    def run(self) -> None:
        """Main conversation loop handling user commands until exit.

        The loop listens to the user's requests, supports commands like
        'show cart' and 'checkout', and otherwise treats input as an item
        to search for and add to the cart.
        """
        self.speak("Hello! I'm your voice shopping assistant. I can help you find and order groceries from Coles, Woolworths, and ALDI.")
        self.speak("What would you like to order today?")

        while True:
            user_input = self.listen()

            # If listening failed, skip iteration and try again
            if not user_input:
                continue

            # Exit commands - stop the assistant
            if any(word in user_input for word in ["exit", "quit", "goodbye", "bye", "stop"]):
                self.speak("Thank you for using the shopping assistant. Goodbye!")
                break

            # Cart inspection commands
            if "show cart" in user_input or "view cart" in user_input or "what's in my cart" in user_input:
                self.show_cart()
                self.speak("Would you like to add more items or checkout?")
                continue

            # Checkout commands
            if "checkout" in user_input or "place order" in user_input or "order now" in user_input:
                if self.checkout():
                    # Offer to continue shopping after a successful order
                    self.speak("Would you like to order anything else?")
                continue

            # Otherwise, treat the utterance as an item request and try to parse it
            item_name, quantity = self.extract_item_and_quantity(user_input)

            # Perform product search (mocked) and present options
            self.speak(f"Searching for {item_name}...")
            products = self.search_products(item_name)

            selected_item = self.present_options(products, item_name, quantity)

            if selected_item:
                # Ask whether to add to cart, order immediately, or cancel
                self.speak("Would you like to add this to your cart, order it now, or cancel?")
                action = self.listen()

                if "cancel" in action or "no" in action or "neither" in action:
                    self.speak("Okay, cancelled.")
                elif "order now" in action or "order it" in action or "immediately" in action:
                    # Add the item and attempt checkout immediately
                    self.cart.append(selected_item)
                    self.checkout()
                else:
                    # Default: add to cart if the user doesn't explicitly order now
                    self.add_to_cart(selected_item)

                # Prompt for additional items
                self.speak("What else would you like to order?")


# Usage example: run the assistant when the module is executed directly
if __name__ == "__main__":
    assistant = VoiceShoppingAssistant()
    assistant.run()