import unittest
import torch
from torch import nn
from EmotionNet import EmotionNet  
import platform

def get_device():
    os_name = platform.system()
    if os_name == "Windows" or os_name == "Linux": #for Win and Liunx
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif os_name == "Darwin": # for Macs
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        return torch.device("cpu")
    
class TestEmotionNet(unittest.TestCase):
    def setUp(self):
        self.model = EmotionNet()
        self.dummy_input = torch.randn(1, 3, 48, 48)  

    def test_model_initialization(self):
        self.assertIsInstance(self.model, EmotionNet)

    def test_forward_pass(self):
        self.model.eval()
        output = self.model(self.dummy_input)
        self.assertEqual(output.shape[0], 1) 

    def test_output_shape(self):
        num_classes = 7  
        output = self.model(self.dummy_input)
        self.assertEqual(output.shape[-1], num_classes)

    def test_model_trainability(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        output = self.model(self.dummy_input)
        target = torch.tensor([0]) 
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_model_to_device(self):
        device = get_device()
        self.model.to(device)
        for param in self.model.parameters():
            self.assertEqual(param.device.type, device.type)
        if hasattr(self.model, 'conv1'):
            self.assertEqual(self.model.conv1.weight.device, device)
            

    def test_gradient_flow(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        output = self.model(self.dummy_input)
        loss = output.mean()  
        loss.backward()
        has_grad = all([param.grad is not None for param in self.model.parameters() if param.requires_grad])
        self.assertTrue(has_grad)

if __name__ == '__main__':
    unittest.main()
