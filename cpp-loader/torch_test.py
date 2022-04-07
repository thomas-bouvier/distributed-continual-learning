import torch
import rehearsal

if __name__ == "__main__":
    l = [("a", torch.FloatTensor([1,2,3])), ("b", torch.FloatTensor([4,5,6]))]
    sl = rehearsal.StreamLoader(3, 10)
    sl.accumulate(l)
