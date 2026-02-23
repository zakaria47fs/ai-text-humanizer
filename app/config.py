import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    PORT: int = int(os.getenv("PORT", "8111"))


settings = Settings()
