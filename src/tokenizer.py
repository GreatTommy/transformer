import unicodedata


class ASCIITokenizer:
    def __init__(self, ascii_range=(32, 127)):
        self.ascii_range = range(ascii_range[0], ascii_range[1])

    def __len__(self):
        return len(self.ascii_range)

    def encode(self, sequence):
        sequence = (
            unicodedata.normalize("NFKD", sequence)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        return [
            ord(char) - min(self.ascii_range)
            for char in sequence
            if ord(char) in self.ascii_range
        ]

    def decode(self, ids):
        return "".join([chr(id + min(self.ascii_range)) for id in ids])
