import spacy
import speech_recognition as sr

class TextToISL:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.recognizer = sr.Recognizer()

    def get_user_input(self):
        print("Choose an input method:")
        print("1. Text")
        print("2. Live Audio")
        print("3. Audio File")
        choice = input("Enter 1 for Text, 2 for Live Audio, or 3 for Audio File: ")

        if choice == "1":
            # Text Input
            inp_sent = input("Enter your text: ")
            return inp_sent

        elif choice == "2":
            # Live Audio Input
            with sr.Microphone() as source:
                print("Listening... Speak into the microphone.")
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    inp_sent = self.recognizer.recognize_google(audio)
                    print(f"Live Audio Input Recognized as: {inp_sent}")
                    return inp_sent
                except sr.UnknownValueError:
                    print("Sorry, could not understand the audio.")
                    return None
                except sr.RequestError as e:
                    print(f"API error: {e}")
                    return None

        elif choice == "3":
            # Audio File Input
            audio_file = input("Enter the path to your audio file: ")
            try:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
                inp_sent = self.recognizer.recognize_google(audio)
                print(f"Audio File Recognized as: {inp_sent}")
                return inp_sent
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the audio.")
                return None
            except sr.RequestError:
                print("Speech Recognition service is unavailable.")
                return None
            except FileNotFoundError:
                print(f"Audio file not found: {audio_file}")
                return None
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            return None

    def lower_case(self, text):
        return text.lower()

    def tokenize(self, text):
        doc = self.nlp(text)
        return [(i, word.text) for i, word in enumerate(doc)]

    def lemmatize(self, text):
        doc = self.nlp(text)
        return [word.lemma_ for word in doc if not word.is_stop]

    def POS(self, text):
        doc = self.nlp(text)
        return [(word.text, word.pos_, word.dep_) for word in doc]

    def convert_to_isl(self, text):
        """Convert the text to ISL format based on grammatical rules."""
        doc = self.nlp(text)

        subject = ""
        verb = ""
        objects = []
        adjectives = []
        question_word = ""
        possessive = ""
        adverbs = []
        punctuation = ""

        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text

            if token.pos_ == "VERB" and not verb:
                verb = token.text

            if token.dep_ in ("dobj", "pobj", "attr", "obj"):
                objects.append(token.text)

            if token.pos_ == "ADJ":
                adjectives.append(token.text)

            if token.tag_ in ("WP", "WRB"):
                question_word = token.text

            if token.dep_ == "poss":
                possessive = token.text

            if token.pos_ == "ADV":
                adverbs.append(token.text)

            if token.pos_ == "PUNCT":
                punctuation = token.text

        isl_sentence = []

        if possessive:
            isl_sentence.append(possessive)

        if subject:
            isl_sentence.append(subject)

        if adjectives:
            isl_sentence.extend(adjectives)

        if objects:
            isl_sentence.extend(objects)

        if adverbs:
            isl_sentence.extend(adverbs)

        if verb:
            isl_sentence.append(verb)

        if punctuation:
            isl_sentence.append(punctuation)

        return " ".join(isl_sentence)

    def process_text(self):
        inp_sent = self.get_user_input()

        if inp_sent:
            lowercased_text = self.lower_case(inp_sent)
            print("Lowercased Text:", lowercased_text)
            print("Tokens:", self.tokenize(lowercased_text))
            print("POS Tags:", self.POS(lowercased_text))
            isl_sentence = self.convert_to_isl(lowercased_text)
            print("ISL Sentence:", isl_sentence)

if __name__ == "__main__":
    text_to_ISL = TextToISL()
    text_to_ISL.process_text()
