import pygame

path = "/home/chikere/Documents/Embedded_Systms_Scope/"
sounds = ["censor-beep-1sec-8112.mp3","censor-beep-10sec-8113.mp3","censor-beep-88052.mp3","emergency-alarm-with-reverb-29431.mp3"]

pygame.mixer.init()
speaker_volume = 0.5
pygame.mixer.music.set_volume(speaker_volume)

for soundfile in sounds:
	pygame.mixer.music.load(path+soundfile)
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy() == True:
		continue



import pygame
import RPi.GPIO as GPIO
import time

# Path to the sound files and the sound file list
path = "/home/chikere/Documents/Embedded_Systms_Scope/"
sounds = [
    "censor-beep-1sec-8112.mp3",
    "censor-beep-10sec-8113.mp3",
    "censor-beep-88052.mp3",
    "emergency-alarm-with-reverb-29431.mp3"
]

# GPIO setup
GPIO.setmode(GPIO.BCM)
LEDs = [17, 18, 22, 23]
for led in LEDs:
    GPIO.setup(led, GPIO.OUT)

# Pygame mixer initialization
pygame.mixer.init()
speaker_volume = 0.5
pygame.mixer.music.set_volume(speaker_volume)

# Function to play sound and light corresponding LED
def play_sound_and_light_led(soundfile, led):
    pygame.mixer.music.load(path + soundfile)
    pygame.mixer.music.play()
    GPIO.output(led, True)  # Turn on LED
    while pygame.mixer.music.get_busy():
        continue
    GPIO.output(led, False)  # Turn off LED

# Main loop to play each sound with corresponding LED
for i, soundfile in enumerate(sounds):
    play_sound_and_light_led(soundfile, LEDs[i])

# Clean up GPIO and quit pygame
GPIO.cleanup()
pygame.quit()
