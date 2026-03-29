import gradio as gr
from src.chat_bot import get_response

def chat(user_input):
    return get_response(user_input)

interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(placeholder="Ask a cricket question..."),
    outputs="text",
    title="🏏 Cricket AI Chatbot",
    description="Ask anything about cricket rules, techniques, or gameplay"
)

if __name__ == "__main__":
    interface.launch()