import requests
import json
import os
import base64
from dotenv import load_dotenv

load_dotenv()

class HuggingChat_RE:
    def __init__(self, hf_chat: str = os.environ.get("HUGGING_CHAT_ID"), model: str = "meta-llama/Meta-Llama-3-70B-Instruct") -> None:
        """
        Initializes an instance of the HuggingChat_RE class.

        Parameters:
        - hf_chat (str): The Hugging Face chat token.
        - model (str): The name or path of the model to be used for the chat. Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".

        Returns:
        - None: This is a constructor method and does not return anything.
        """

        self.hf_chat = hf_chat
        self.model = model
        self.headers = {
            "Cookie": f"hf-chat={self.hf_chat}",
        }
        self.conversationId = self.find_conversation_id()
        self.messageId = self.find_message_id()

    def find_conversation_id(self) -> str:
        """
        Finds and returns the conversation ID for the Hugging Face chat.

        Returns:
        - str: The conversation ID retrieved from the server response.
        """

        url = "https://huggingface.co/chat/conversation"
        payload = {"model": self.model}
        response = requests.post(url, json=payload, headers=self.headers).json()
        print("\033[92m" + "Initialised Conversation ID:", response['conversationId'] + "\033[0m")
        return response['conversationId']

    def find_message_id(self) -> str:
        """
        Finds and returns the message ID for the Hugging Face chat.

        Returns:
        - str: The message ID retrieved from the server response.
        """

        url = f"https://huggingface.co/chat/conversation/{self.conversationId}/__data.json?x-sveltekit-invalidated=11"
        response = requests.get(url, headers=self.headers).json()
        print("\033[92m" + "Initialised Message ID:", response['nodes'][1]['data'][3] + "\033[0m")
        return response['nodes'][1]['data'][3]
    
    def download_image(self, sha_value, output_filename="downloaded_image.png"):
        # Construct the image URL
        image_url = f"https://huggingface.co/chat/conversation/{self.conversationId}/output/{sha_value}"
        
        # Define headers with required cookies and user-agent
        headers = {
            "Cookie": "hf-chat=94bb815b-befa-4f2a-b194-c7fff7c1b012",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        try:
            # Send a GET request to the image URL with headers
            response = requests.get(image_url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                # Open a file in binary write mode
                with open(output_filename, "wb") as file:
                    # Write the image data to the file
                    file.write(response.content)
                print("\033[92m" + f"Image successfully downloaded and saved as {output_filename}" + "\033[0m")
                return output_filename
            else:
                print(f"Failed to download image. Status code: {response.status_code}")

        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    def generate(self, query: str, web_search: bool = False, filepath: str = None, stream: bool = True, output_filename: str = None) -> str:
        """
        Generates a response for the given query using the Hugging Face chat.

        Parameters:
        - query (str): The text query to generate a response for.
        - web_search (bool): Flag for web search. Defaults to False.
        - filepath (str): Path to the file. Defaults to None.
        - stream (bool): Flag for streaming response. Defaults to True.

        Returns:
        - str: The complete response.
        """

        url = f"https://huggingface.co/chat/conversation/{self.conversationId}"

        files_to_send = []
        if filepath is not None:
            with open(filepath, "rb") as file:
                base64_content = base64.b64encode(file.read()).decode("utf-8")

            files_to_send = [{
                "mime": "application/pdf",  # Adjust if necessary
                "name": os.path.basename(filepath),
                "type": "base64",
                "value": base64_content
            }]

        payload = {
            "inputs": query,
            "id": self.messageId,
            "is_retry": False,
            "is_continue": False,
            "web_search": web_search,
            "files": files_to_send 
        }

        response = requests.post(url, json=payload, headers=self.headers, stream=True)
        complete_response = ""
        for chunk in response.iter_lines(chunk_size=1, decode_unicode=True):
            if chunk:
                try:
                    json_data = json.loads(chunk.strip())
                    if json_data['type'] == "stream":
                        if stream: print(json_data['token'], end="", flush=True)
                        complete_response += json_data['token']
                        
                    elif json_data['type'] == "tool" and json_data['subtype'] != "result":
                        print("\033[95m" + "Tools Are Used" + "\033[0m")

                        if json_data['subtype'] == "call":
                            print("\033[93m" + "Tools Name:", json_data['call']['name'] + "\033[0m")
                            print("\033[93m" + "Image Prompt:", json_data['call']['parameters']['prompt'] + "\033[0m", "\n")

                        elif json_data['subtype'] == "result":
                            print("\033[93m" + "Tools Result:", json_data['result'] + "\033[0m", "\n")

                    elif json_data['type'] == "file":
                        print("\033[95m" + "File Details" + "\033[0m")
                        print("\033[93m" + "Name:", json_data['name'] + "\033[0m")
                        print("\033[93m" + "Sha:", json_data['sha'] + "\033[0m")
                        if output_filename: self.download_image(json_data['sha'], output_filename)
                        else: self.download_image(json_data['sha'])

                except:
                    continue
        
        return complete_response

# Example Usage
if __name__ == "__main__":

    hf_api = HuggingChat_RE(model="CohereForAI/c4ai-command-r-plus")
    while True:
        query = input("\n> ")
        response = hf_api.generate(query, web_search=False)
        
        # response = hf_api.generate(query, web_search=False, filepath=r"C:\Users\sreej\Downloads\about_blank (2).pdf")
        # response = hf_api.generate(query, stream=False)
        # print(response)

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
        
        hf_api = HuggingChat_RE(model=name)
        hf_api.generate("Who are you ?")


"******************************************************************"

"""Deprecated v1.0. Lack Support of Tools Support For Cohere. Still works For Other Models"""

"******************************************************************"


# import requests
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class HuggingChat_RE:
#     def __init__(self, hf_chat: str = os.environ.get("HUGGING_CHAT_ID"), model: str = "meta-llama/Meta-Llama-3-70B-Instruct") -> None:
#         """
#         Initializes an instance of the HuggingChat_RE class.

#         Parameters:
#         - hf_chat (str): The Hugging Face chat token.
#         - model (str): The name or path of the model to be used for the chat. Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".

#         Returns:
#         - None: This is a constructor method and does not return anything.
#         """

#         self.hf_chat = hf_chat
#         self.model = model
#         self.headers = {
#             "Cookie": f"hf-chat={self.hf_chat}",
#         }
#         self.conversationId = self.find_conversation_id()
#         self.messageId = self.find_message_id()

#     def find_conversation_id(self) -> str:
#         """
#         Finds and returns the conversation ID for the Hugging Face chat.

#         Returns:
#         - str: The conversation ID retrieved from the server response.
#         """

#         url = "https://huggingface.co/chat/conversation"
#         payload = {"model": self.model}
#         response = requests.post(url, json=payload, headers=self.headers).json()
#         print("\033[92m" + "Initialised Conversation ID:", response['conversationId'] + "\033[0m")
#         return response['conversationId']

#     def find_message_id(self) -> str:
#         """
#         Finds and returns the message ID for the Hugging Face chat.

#         Returns:
#         - str: The message ID retrieved from the server response.
#         """

#         url = f"https://huggingface.co/chat/conversation/{self.conversationId}/__data.json?x-sveltekit-invalidated=11"
#         response = requests.get(url, headers=self.headers).json()
#         print("\033[92m" + "Initialised Message ID:", response['nodes'][1]['data'][3] + "\033[0m")
#         return response['nodes'][1]['data'][3]

#     def generate(self, query: str, web_search: bool = False, files=[], stream: bool = True) -> str:
#         """
#         Generates a response for the given query using the Hugging Face chat.

#         Parameters:
#         - query (str): The text query to generate a response for.
#         - web_search (bool): A flag indicating whether to perform web search in the response generation process. Defaults to False.
#         - files (List[str]): A list of file paths to include in the query. Defaults to an empty list.
#         - stream (bool): A flag indicating whether to stream the response. Defaults to True.

#         Returns:
#         - str: The complete response generated by the chat.
#         """

#         url = f"https://huggingface.co/chat/conversation/{self.conversationId}"
#         payload = {
#             "inputs": query,
#             "id": self.messageId,
#             "is_retry": False,
#             "is_continue": False,
#             "web_search": web_search,
#             "files": files
#         }


#         response = requests.post(url, json=payload, headers=self.headers, stream=True)
#         complete_response = ""
#         for chunk in response.iter_content(chunk_size=1024):
#             if chunk:
#                 # print(chunk)
#                 try:
#                     json_data = json.loads(chunk.decode("utf-8"))
#                     if json_data['type'] == "stream":
#                         if stream: print(json_data['token'], end="", flush=True)
#                         complete_response += json_data['token']
#                 except:
#                     continue

#         return complete_response

# # Example Usage
# if __name__ == "__main__":

#     hf_api = HuggingChat_RE(model="CohereForAI/c4ai-command-r-plus")
#     while True:
#         query = input("\n> ")
#         response = hf_api.generate(query, web_search=False)
#         # response = hf_api.generate(query, stream=False)
#         # print(response)

#     models = {
#     "meta-llama/Meta-Llama-3-70B-Instruct": "https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
#     "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": "https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
#     "CohereForAI/c4ai-command-r-plus": "https://huggingface.co/CohereForAI/c4ai-command-r-plus",
#     "mistralai/Mixtral-8x7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
#     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
#     "google/gemma-1.1-7b-it": "https://huggingface.co/google/gemma-1.1-7b-it",
#     "mistralai/Mistral-7B-Instruct-v0.2": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
#     "microsoft/Phi-3-mini-4k-instruct": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"
#     }

#     for name, url in models.items():
#         print("\n\n" + name)
        
#         hf_api = HuggingChat_RE(model=name)
#         hf_api.generate("Who are you ?")
