import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):        
        super().__init__()        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        #Architecture with layers embed, lstm and linear.
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)                        
    
    def forward(self, features, captions):
        
        #Concatenating the features and captions and pass through the network.
        embeds = self.embed(captions[:,:-1])        
        embeds = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)        
        lstm_out, lstm_state = self.lstm(embeds, None)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):

        caption = []
        
        # initialize lstm state
        lstm_state = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        # Get the caption
        for i in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            outputs = self.linear(lstm_out)       
            outputs = outputs.squeeze(1)                
            word = outputs.argmax(dim=1)            
            caption.append(word.item())
            
            inputs = self.embed(word.unsqueeze(0))
          
        return caption
