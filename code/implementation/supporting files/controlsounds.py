import pygame
import RPi.GPIO as GPIO
import time
from sys import argv

# Checking if enough arguments are provided
if len(argv) < 3:
    print("Usage: script.py <whichled> <action>")
    sys.exit(1)

whichled = argv[1]
action = argv[2]

# Path to the sound files and the sound file list
path = "/home/chikere/Documents/Embedded_Systms_Scope/"
sounds = {
    'a': "censor-beep-1sec-8112.mp3",
    'b': "censor-beep-10sec-8113.mp3",
    'c': "censor-beep-88052.mp3",
    'd': "emergency-alarm-with-reverb-29431.mp3",
}

# GPIO setup
GPIO.setmode(GPIO.BCM)
LEDs = {'a': 17, 'b': 18, 'c': 22, 'd': 23}
for led in LEDs.values():
    GPIO.setup(led, GPIO.OUT)

# Pygame mixer initialization
pygame.mixer.init()
speaker_volume = 0.5
pygame.mixer.music.set_volume(speaker_volume)

# Function to control all LEDs except the selected one
def control_other_leds(exclude, state):
    for key, led in LEDs.items():
        if key != exclude:
            GPIO.output(led, state)

# Play sound and control LED
def play_sound_and_control_led(led, soundfile, action):
    control_other_leds(whichled, False)  # Turn off all other LEDs
    if action == "on":
        pygame.mixer.music.load(path + soundfile)
        pygame.mixer.music.play()
        GPIO.output(led, True)  # Turn on the selected LED
        while pygame.mixer.music.get_busy():
            continue
        GPIO.output(led, False)  # Optionally turn off the LED after sound has played
    elif action == "off":
        GPIO.output(led, False)  # Turn off the selected LED

# Determine the LED and sound to control
if whichled in LEDs:
    play_sound_and_control_led(LEDs[whichled], sounds[whichled], action)
else:
    print("Invalid LED identifier")

# Clean up GPIO and quit pygame
GPIO.cleanup()
pygame.quit()
