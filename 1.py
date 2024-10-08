import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download

# Установим параметры для кеширования файлов вручную (если нужно)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Отключаем предупреждение о symlink на Windows

# Загружаем модель
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    use_auth_token="ваш_токен_здесь"
)

# Запрос на генерацию изображения
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt).images[0]

# Сохранение изображения
image.save("flux-dev.png")

# #Аутентификация в терминале:
# huggingface-cli login
# #После этого вам нужно будет ввести ваш токен. Это аутентифицирует вашу среду для доступа к закрытым репозиториям, и diffusers сможет использовать этот токен автоматически.