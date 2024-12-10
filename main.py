import tkinter as tk
from tkinter import ttk, scrolledtext
from LLama_index import get_top_matches, initialize_index, extract_preferences_with_gpt
from SNKER_v1 import analyze_sentiment, generate_personalized_recommendation
import time

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sneaker Recommendation Chatbot")
        self.root.geometry("600x400") 
        
        self.index = initialize_index()

        self.chat_history = ""
        

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        

        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame, 
            wrap=tk.WORD, 
            width=50, 
            height=20
        )
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            self.main_frame, 
            textvariable=self.input_var,
            width=40
        )
        self.input_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.send_button = ttk.Button(
            self.main_frame, 
            text="Send", 
            command=self.send_message
        )
        self.send_button.grid(row=1, column=1, sticky=(tk.E))
        
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        

        self.display_message("Bot: Welcome! Please tell me what kind of sneakers you're looking for.")

    def send_message(self):
        user_input = self.input_var.get().strip()
        if not user_input:
            return
        
        self.display_message(f"You: {user_input}")
        self.input_var.set("")

        # Get flag to check if user wants to end conversation
        _, _, flag = get_top_matches(user_input, self.chat_history, self.index)
        
        # If user wants to end conversation, display farewell message and close
        if int(flag[0]):
            self.display_message("Bot: I'm always glad to help!")
            time.sleep(1)
            self.root.quit()
            return
        
        # Only show searching message if user wants to continue
        self.display_message("Bot: Got it, I'm searching for you...", tag="searching")
        self.root.update_idletasks()
        
        self.process_input(user_input)

    def process_input(self, user_input):
        if user_input.lower() == 'quit':
            self.root.quit()
            return
        
        self.chat_history = self.chat_history + user_input + " "
        print('History: ',self.chat_history)
            
        sentiment = analyze_sentiment(user_input)
        top_matches, sum_userinput, flag = get_top_matches(user_input, self.chat_history, self.index)
        print(flag)

        self.chat_display.delete("searching.first", "searching.last")
        
        if not top_matches:
            self.display_message("Bot: Sorry, I couldn't find any matching sneakers.")
            return
        
        best_match = top_matches[0]
        recommendation = generate_personalized_recommendation(
            sum_userinput, best_match, sentiment
        )
        
        self.display_message(f"Bot: {recommendation} \n\nPlease let me know if you have other preference.")

    def display_message(self, message, tag=None):
        self.chat_display.insert(tk.END, message + "\n\n", tag)
        self.chat_display.see(tk.END) 
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 