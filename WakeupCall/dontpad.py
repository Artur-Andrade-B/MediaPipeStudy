import pygame
pygame.mixer.init()
def play_alarm():
    pygame.mixer.music.load("WakeupCall/audio/alarm1.mp3", "alarme")
    pygame.mixer.music.play()
def stop_alarm():
    pygame.mixer.music.stop()
run = True

while run:
    sel = input("inicia ou para")
    if sel == "1":
        play_alarm()
    else:
        run = False
        stop_alarm