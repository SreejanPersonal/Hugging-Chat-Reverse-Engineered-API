<div align="center">
  <img src="https://img.shields.io/badge/HuggingChat-API-blue?style=for-the-badge&logo=huggingface" alt="HuggingChat API Badge">
  <h1>HuggingChat API 🤗🤗 - Unofficial Reverse Engineering 🚀</h1>
  <p>
    <a href="https://github.com/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API?style=social">
    </a>
    <a href="https://github.com/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API/network/members">
      <img alt="GitHub forks" src="https://img.shields.io/github/forks/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API?style=social">
    </a>
    <a href="https://github.com/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API/issues">
      <img alt="GitHub issues" src="https://img.shields.io/github/issues/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API?style=social">
    </a>
  </p>
</div>

<div align="center">
  <a href="https://youtube.com/@devsdocode"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://t.me/devsdocode"><img alt="Telegram" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  <a href="https://www.instagram.com/sree.shades_/"><img alt="Instagram" src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/developer-sreejan/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="https://buymeacoffee.com/devsdocode"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black"></a>
</div>

## Crafted with ❤️ by Devs Do Code (Sree)

> **Disclaimer:** This project is not officially associated with Hugging Face. It is an independent reverse engineering effort to explore the Hugging Chat API.

---

## 🚀 Repository Status Update

🛑 **Important Notice:** 
This repository is no longer maintained by the owner `Devs Do Code (Sree)`. Any contributions to this repository are heartily welcomed 💝💝.

---

## 📜 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)
- [Get in Touch](#-get-in-touch)

---

### 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SreejanPersonal/Hugging-Chat-Reverse-Engineered-API.git
   ```
2. **Navigate to the Hugging Chat platform and sign in with your Hugging Face account.** 
   (Avoid guest mode for full functionality.)
3. **Choose any available model on Hugging Chat to initiate a new conversation.**
4. **Access Developer Tools with `Ctrl + Shift + I` and select the `Network` tab.**
5. **Engage with the model by entering a query in the chat interface.**
6. **Observe the API requests in the `Network` tab and locate the `conversation` request.**
7. **In the `Response Headers` section, find the `Set-Cookie` entry and copy the `hf-chat` value.**
   Treat this as your API key and keep it confidential.

---

### 🛠️ Installation

After cloning the repository and obtaining your `HF_CHAT_ID`, proceed as follows:

1. **Navigate to the `Hugging-Chat-Reverse-Engineered-API` directory:**
   ```bash
   cd Hugging-Chat-Reverse-Engineered-API
   ```
2. **Create a `.env` file and store your `HF_CHAT_ID` within:**
   ```bash
   echo "HF_CHAT_ID=your_hf_chat_id" > .env
   ```
3. **Execute `main.py` to start interacting with the API:**
   ```bash
   python main.py
   ```

---

### 💻 Usage

To utilize the API, ensure you have the necessary dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

- **Running the script:**
  ```bash
  python main.py
  ```

- **Make sure to replace `your_hf_chat_id` with the actual `hf-chat` value obtained from the response headers.**

---

### 🤝 Contributing

Your contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

### 📜 License

This project is licensed under the [MIT License](LICENSE). Full license text is available in the [LICENSE](LICENSE) file.

---

### 📬 Get in Touch

For inquiries or assistance, please open an issue or reach out through our social channels:

<div align="center">
  <a href="https://youtube.com/@devsdocode"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://t.me/devsdocode"><img alt="Telegram" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  <a href="https://www.instagram.com/sree.shades_/"><img alt="Instagram" src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/developer-sreejan/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="https://buymeacoffee.com/devsdocode"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black"></a>
</div>

We appreciate your interest in `Hugging-Chat-Reverse-Engineered-API`.
