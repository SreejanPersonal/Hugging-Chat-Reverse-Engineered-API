# ██████╗   ███████╗ ██╗    ██╗ ███████╗            ██████╗    ██████╗             ██████╗   ██████╗    ██████╗   ███████╗
# ██╔══██╗ ██╔════╝ ██║    ██║ ██╔════╝            ██╔══██╗ ██╔═══██╗          ██╔════╝ ██╔═══██╗ ██╔══██╗ ██╔════╝
# ██║    ██║ █████╗     ██║    ██║ ███████╗            ██║    ██║ ██║      ██║         ██║          ██║      ██║ ██║    ██║ █████╗  
# ██║    ██║ ██╔══╝    ╚██╗  ██╔╝ ╚════██║           ██║    ██║ ██║      ██║         ██║          ██║      ██║ ██║    ██║ ██╔══╝  
# ██████╔╝ ███████╗  ╚████╔╝  ███████║            ██████╔╝╚██████╔╝          ╚██████╗ ╚██████╔╝  ██████╔╝ ███████╗
# ╚═════╝  ╚══════╝    ╚═══╝     ╚══════╝            ╚═════╝    ╚═════╝             ╚═════╝   ╚═════╝    ╚═════╝   ╚══════╝

#  Made With 💓 By - Sree ( Devs Do Code )
#  YouTube Channel: https://www.youtube.com/@devsdocode

#  For any questions or concerns, reach out to us via our social media handles.
#  Our top choice for contact is Telegram: https://t.me/devsdocode
#  You can also find us on other platforms listed above. We're here to help!

#  - YouTube Channel: https://www.youtube.com/@DevsDoCode
#  - Telegram Group: https://t.me/devsdocode
#  - Discord Server: https://discord.gg/ehwfVtsAts
#  - Instagram:
#    - Personal: https://www.instagram.com/sree.shades_/
#    - Channel: https://www.instagram.com/devsdocode_/

#  ------------------------------------------------------------------------------
#  Dive into the world of coding with Devs Do Code - where passion meets programming!
#  Make sure to hit that Subscribe button to stay tuned for exciting content!

#  Pro Tip: For optimal performance and a seamless experience, we recommend using
#  the default library versions demonstrated in this demo. Your coding journey just
#  got even better! Happy coding!
#  ----------------------------------------------------------------------------

from Hugging_Chat import HuggingChat_RE
import os
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    
    models = {
    "meta-llama/Meta-Llama-3-70B-Instruct": "https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": "https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "CohereForAI/c4ai-command-r-plus": "https://huggingface.co/CohereForAI/c4ai-command-r-plus",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "google/gemma-1.1-7b-it": "https://huggingface.co/google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-Instruct-v0.2": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"
    }

    for name, url in models.items():
        print("\n\n" + name)
        
        hf_api = HuggingChat_RE(model=name, hf_chat=os.environ.get("HUGGING_CHAT_ID"))
        hf_api.generate("Who are you ?")