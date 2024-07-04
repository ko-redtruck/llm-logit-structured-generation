import unittest
from nodes import DenseLogitsTokenNode
from transformers import AutoTokenizer

"""
Maybe the test cases shouldn't hard code the input_ids and instead use self.tokenizer in each step.
This way changing the tokenizer would not break all test cases!
"""

class TestEmptyTokenNode(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    
    def test_graph_with_multiple_paths_from_text(self):
        graph = DenseLogitsTokenNode.from_text_without_logits("Hello World", self.tokenizer) # 8774, 1150, 1
        graph.add_text_without_logits("Hello Fox", self.tokenizer) #8774 7547 1
        graph.add_text_without_logits("Wow!!", self.tokenizer) #9758, 1603, 1

        #Hello and Wow
        self.assertCountEqual(graph.nexts.keys(), [8774, 9758])

        hello_node = graph.nexts[8774]
        #world and fox
        self.assertCountEqual(hello_node.nexts.keys(), [7547, 1150])

        self.assertListEqual(graph.gather_all_sequences(), [[8774, 1150, 1], [8774, 7547, 1], [9758, 1603, 1]])
        
    def test_from_text(self):
        text = "Hello world" #8774, 296, 1

        graph = DenseLogitsTokenNode.from_text_without_logits(text, self.tokenizer)
        self.assertTrue(graph.is_start_token_node())

        next_node = graph.nexts[8774]
        self.assertEqual(next_node.token_id, 8774)

        next_node = next_node.nexts[296]
        self.assertEqual(next_node.token_id, 296)

        last_node = next_node.nexts[1]
        self.assertEqual(last_node.token_id, 1)

        print(last_node.nexts)
        self.assertTrue(len(last_node.nexts) == 0)
      
    def test_gather_all_sequences(self):
        text = "Hello world" #8774, 296, 1
        graph = DenseLogitsTokenNode.from_text_without_logits(text, self.tokenizer)

        sequences = graph.gather_all_sequences()
        self.assertListEqual(sequences, [[8774, 296, 1]])

    def test_create_encoder_sequence(self):
        text = "Hello world" #8774, 296, 1
        graph = DenseLogitsTokenNode.from_text_without_logits(text, self.tokenizer, is_encoder_sequence=True)

        sequences = graph.gather_all_sequences()
        self.assertListEqual(sequences, [[8774, 296, 1]])
        self.assertEqual(graph.gather_encoder_sequence(), [8774, 296, 1])

    """
    Is this the right approach? Should there maybe be two different graphs for encoder/decoder or should it be called input/label
    """
    def test_encoder_decoder_pairs(self):
        text = "Hello world" #8774, 296, 1

        graph = DenseLogitsTokenNode.from_text_without_logits(text, self.tokenizer, is_encoder_sequence=True)

        encoder_input_end_node = graph.find_encoder_input_end_node()
        encoder_input_end_node.add_text_without_logits("Hello", self.tokenizer)
        encoder_input_end_node.add_text_without_logits("Goodbye", self.tokenizer)

        encoder_tokens, decoder_tokens = graph.gather_all_encoder_decoder_sequence_pairs()

        self.assertListEqual(encoder_tokens, [[8774, 296, 1], [8774, 296, 1]])
        self.assertListEqual(decoder_tokens, [[8774, 1], [1804, 969, 15, 1]])