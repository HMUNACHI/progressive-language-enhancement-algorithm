
import spacy
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class PLEA:
    def __init__(self, 
                 text_processing_model="en_core_web_sm",
                 text_embedding_model='sentence-transformers/all-MiniLM-L6-v2', 
                 mask_filling_model='distilroberta-base'):
        
        self.text_parser = spacy.load(text_processing_model)
        self.sentence_embedding_model = SentenceTransformer(text_embedding_model)
        self.mask_filling_model = pipeline('fill-mask', model=mask_filling_model)
        self.tags_to_enhance = set(["JJR", "PDT", "RBR"])
        self.tags_to_replace = set(["MD"])
        self.enhanced_texts = []


    def enhance_text(self, text, top_k=1, confidence=0.2, tolerance=0.9):
        self.enhanced_texts.append(text)
        tags = self.get_granular_parts_of_speech(text)

        for index in range(len(tags)):
            if tags[index] in self.tags_to_enhance:
                masked_text = self.insert_mask(text, index)
            elif tags[index] in self.tags_to_replace:
                masked_text = self.replace_word_with_mask(text, index)
            else:
                continue

            candidates = self.fill_mask(masked_text, top_k, confidence)
            if len(candidates) < 1:
                continue

            similarity_scores = self.batch_sentence_similarities(text, candidates)
            self.extend_enhanced_texts(candidates, similarity_scores, tolerance)

        return self.enhanced_texts
    
    
    def get_granular_parts_of_speech(self, text):
        processed_text = self.text_parser(text)
        return [token.tag_ for token in processed_text]


    def insert_mask(self, text, index):
        split_text = text.split(" ")
        return " ".join(split_text[:index] + ["<mask>"] + split_text[index:])

    
    def replace_word_with_mask(self, text, index, n_masks=1):
        split_text = text.split(" ")
        masks = ["<mask>" for _ in range(n_masks)]
        return " ".join(split_text[:index] + masks + split_text[index+1:])


    def fill_mask(self, masked_text, top_k, confidence):
        outputs = self.mask_filling_model(masked_text)[:top_k]
        confident_outputs = [output['sequence'] for output in outputs if output["score"]>=confidence]
        return confident_outputs


    def batch_sentence_similarities(self, source, candidates):
        source_embedding = self.sentence_embedding_model.encode(source)
        candidates_embeddings = self.sentence_embedding_model.encode(candidates)
        return self.batch_cosine_similarities(source_embedding, candidates_embeddings)


    def batch_cosine_similarities(self, source, candidates):
        dot_products = np.einsum("ij,j->i", candidates, source)
        norm_source = np.sqrt(np.einsum('i,i->', source, source))
        norm_candidates = np.sqrt(np.einsum('ij,ij->i', candidates, candidates))
        return dot_products / (norm_source * norm_candidates)


    def extend_enhanced_texts(self, candidates, simiarity_scores, tolerance):
        for candidate, score in zip(candidates, simiarity_scores):
            if score >= tolerance:
                self.enhanced_texts.append(candidate)
