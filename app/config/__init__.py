from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
   
    GOOGLE_API_KEY: str
    PGVECTOR_CONNECTION_STRING: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()