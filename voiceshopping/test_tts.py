from voice import VoiceInterface

if __name__ == '__main__':
    vi = VoiceInterface({"voice": {"rate": 150, "volume": 1.0}})
    texts = ["Hello", "This is a second message", "And a third one"]
    for t in texts:
        print('Speaking:', t)
        vi.speak(t)
    print('Done')
