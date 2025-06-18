import pygame

# mp3 파일 한번만 재생
# pygame이 초기화되지 않았다면 초기화
def play_beep(beep_path):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(beep_path)
        pygame.mixer.music.play()