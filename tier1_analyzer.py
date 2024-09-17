import datetime
import glob
import os
import spacy
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from spacy.util import minibatch, compounding
from spacy.training import Example
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    drug: str
    initial_dosage: str
    initial_frequency: str
    dosage_route: str
    conditional_instructions: Optional[str]
    dosage_changes: List[Dict[str, str]]
    duration: Optional[str]
    entities: Dict[str, list]
    confidence: float
    context: Dict[str, Any]

class AdvancedDrugAnalyzer:
    def __init__(self):
        # Load en_core_med7_lg model
        self.nlp = spacy.load("en_core_med7_lg")
        
        # Load drug knowledge base from the provided JSON
        self.drug_kb = self.load_drug_knowledge_base()
        
        # Initialize TF-IDF vectorizer for context similarity
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.drug_kb['response'])
        
        # Build knowledge graph
        self.knowledge_graph = self.build_knowledge_graph()

    def load_drug_knowledge_base(self) -> pd.DataFrame:
        json_data = [
            {
                "prompt": "Patient is a 45-year-old male with a history of high blood pressure and high cholesterol. He is experiencing symptoms of a cold, including a fever and cough. He is also experiencing digestive issues, including heartburn and bloating. The patient is taking medication for his high blood pressure and high cholesterol, but is concerned about the cost of his medications and is looking for ways to save on healthcare.",
                "response": "Medication:\n- Take acetaminophen (Tylenol) 650mg orally every 4-6 hours as needed for fever reduction.\n- Take omeprazole (Prilosec) 20mg orally once daily for heartburn and bloating.\n- Continue taking atorvastatin (Lipitor) 20mg orally once daily for high cholesterol.\n- Continue taking lisinopril (Zestril) 10mg orally once daily for high blood pressure.\n\nInstructions:\n- Drink plenty of fluids to help thin out mucus and soothe a sore throat.\n- Rest and avoid strenuous activities.\n- If symptoms persist or worsen, follow up with your doctor in 3-4 days.\n- If you experience difficulty breathing or chest pain, seek immediate medical attention.\n\nNotes:\n- Consider using GoodRx to compare prices and find discounts on your medications.\n- Discuss any financial concerns with your doctor, as there may be alternative treatment options or resources available to help with medication affordability.\n- Schedule a follow-up appointment with your doctor in 2 weeks to monitor your condition and adjust treatment as needed.",
                "source": "glaive",
                "category": "Dermatological Prescriptions",
                "category_score": 1.0,
                "topic": "Prescription Guidelines for Skin Conditions",
                "topic_score": 0.06985564529895782
            },
            # ... (other JSON data entries)
        ]
        return pd.DataFrame(json_data)

    def build_knowledge_graph(self) -> nx.Graph:
        G = nx.Graph()
        for _, row in self.drug_kb.iterrows():
            response = row['response']
            drugs = re.findall(r'(\w+)\s*\((\w+)\)', response)
            for drug, brand in drugs:
                G.add_node(drug.lower(), type='drug')
                G.add_node(brand.lower(), type='brand')
                G.add_edge(drug.lower(), brand.lower(), type='is_brand_of')
            
            # Add edges between drugs mentioned in the same prescription
            drugs = [drug.lower() for drug, _ in drugs]
            for i in range(len(drugs)):
                for j in range(i+1, len(drugs)):
                    G.add_edge(drugs[i], drugs[j], type='co_prescribed')
        
        return G

    def extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities

    def standardize_entity(self, entity: str, entity_type: str) -> Tuple[str, float]:
        if entity_type == 'DRUG':
            for _, row in self.drug_kb.iterrows():
                if entity.lower() in row['response'].lower():
                    match = re.search(r'(\w+)\s*\((\w+)\)', row['response'])
                    if match:
                        return match.group(1), 1.0
        elif entity_type in ['DOSAGE_ROUTE', 'FREQUENCY']:
            return entity, 0.8
        return entity, 0.5

    def get_context(self, drug: str) -> Dict[str, Any]:
        context = {}
        for _, row in self.drug_kb.iterrows():
            if drug.lower() in row['response'].lower():
                context['category'] = row['category']
                context['topic'] = row['topic']
                context['response'] = row['response']
                break
        return context

    def calculate_confidence(self, entities: List[Dict[str, Any]]) -> float:
        if not entities:
            return 0.0
        return sum(1.0 for entity in entities if entity['label'] in ['DRUG', 'DOSAGE_ROUTE', 'FREQUENCY']) / len(entities)

    def find_similar_prescriptions(self, text: str, top_n: int = 5) -> List[str]:
        text_vector = self.tfidf_vectorizer.transform([text])
        similarities = cosine_similarity(text_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.drug_kb.iloc[top_indices]['response'].tolist()

    def analyze(self, prescription: str) -> AnalysisResult:
        try:
            doc = self.nlp(prescription)
            logger.info(f"Processed text: {doc.text}")
            
            entities = self.extract_entities(doc)
            logger.info(f"Extracted entities: {entities}")
            
            standardized_entities = {}
            for entity in entities:
                standardized, score = self.standardize_entity(entity['text'], entity['label'])
                if entity['label'] not in standardized_entities or score > standardized_entities[entity['label']][1]:
                    standardized_entities[entity['label']] = (standardized, score)
            logger.info(f"Standardized entities: {standardized_entities}")
            
            drug = standardized_entities.get('DRUG', ('', 0))[0]
            initial_dosage, initial_frequency = self.extract_initial_dosage_and_frequency(doc)
            dosage_route = standardized_entities.get('DOSAGE_ROUTE', ('', 0))[0]
            
            conditional_instructions = self.extract_conditional_instructions(doc)
            dosage_changes = self.extract_dosage_changes(doc)
            duration = self.extract_duration(doc)
            
            context = self.get_context(drug) if drug else {}
            
            confidence = self.calculate_confidence(entities)
            
            similar_prescriptions = self.find_similar_prescriptions(doc.text)
            
            return AnalysisResult(
                drug=drug,
                initial_dosage=initial_dosage,
                initial_frequency=initial_frequency,
                dosage_route=dosage_route,
                conditional_instructions=conditional_instructions,
                dosage_changes=dosage_changes,
                duration=duration,
                entities={e['label']: e['text'] for e in entities},
                confidence=confidence,
                context={**context, 'similar_prescriptions': similar_prescriptions}
            )
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return AnalysisResult(
                drug='',
                initial_dosage='',
                initial_frequency='',
                dosage_route='',
                conditional_instructions=None,
                dosage_changes=[],
                duration=None,
                entities={},
                confidence=0.0,
                context={'error': str(e)}
            )

    def extract_initial_dosage_and_frequency(self, doc):
        dosage = ""
        frequency = ""
        for ent in doc.ents:
            if ent.label_ == "DOSAGE" and not dosage:
                dosage = ent.text
            elif ent.label_ == "FREQUENCY" and not frequency:
                frequency = ent.text
        return dosage, frequency

    def extract_conditional_instructions(self, doc):
        conditional_patterns = [
            r"if.*then",
            r"when.*then",
            r"in case of.*",
            r"should.*occur"
        ]
        text = doc.text.lower()
        for pattern in conditional_patterns:
            match = re.search(pattern, text)
            if match:
                return doc.text[match.start():].strip()
        return None

    def extract_dosage_changes(self, doc):
        changes = []
        change_patterns = [
            (r"increase to (\d+(?:\.\d+)?(?:mg|g|ml))", "increase"),
            (r"decrease to (\d+(?:\.\d+)?(?:mg|g|ml))", "decrease"),
            (r"change to (\d+(?:\.\d+)?(?:mg|g|ml))", "change")
        ]
        text = doc.text.lower()
        for pattern, change_type in change_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                changes.append({
                    "type": change_type,
                    "new_dosage": match.group(1),
                    "condition": self.extract_change_condition(text[:match.start()])
                })
        return changes

    def extract_change_condition(self, text):
        time_patterns = [
            r"after (\d+) (day|week|month)",
            r"in (\d+) (day|week|month)"
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1)} {match.group(2)}(s)"
        return "Not specified"

    def extract_duration(self, doc):
        duration_pattern = r"for (\d+) (day|week|month)"
        match = re.search(duration_pattern, doc.text.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}(s)"
        return None

    def get_drug_interactions(self, drug: str) -> List[str]:
        interactions = []
        drug_lower = drug.lower()
        if drug_lower in self.knowledge_graph:
            for neighbor in self.knowledge_graph.neighbors(drug_lower):
                if self.knowledge_graph.nodes[neighbor]['type'] == 'drug':
                    interactions.append(neighbor)
        return interactions

    def continuous_training(self, training_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], iterations: int = 30):
        """
        Continuously train the spaCy model with new data.
        
        :param training_data: List of tuples, each containing (text, {"entities": [(start, end, label)]})
        :param iterations: Number of training iterations
        """
        logger.info("Starting continuous training...")

        # Clear older training data files
        self.clear_old_training_files()

        # Convert training data to spaCy's format
        train_data = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            train_data.append(example)

        # Create a new pipe if 'ner' doesn't exist
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")

        # Add labels
        for example in train_data:
            for ent in example.reference.ents:
                ner.add_label(ent.label_)

        # Get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        # Only train NER
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.initialize()
            
            # Training loop
            for itn in range(iterations):
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    try:
                        self.nlp.update(
                            batch,
                            drop=0.5,
                            losses=losses,
                        )
                    except ValueError as e:
                        if "E203" in str(e):
                            logger.warning("Tok2vec error encountered. Skipping this batch.")
                            continue
                        else:
                            raise e
                logger.info(f"Iteration {itn + 1}, Losses: {losses}")

        # Save the trained model
        self.save_trained_model()

        logger.info("Continuous training completed.")

    def clear_old_training_files(self):
        """Clear older training data files."""
        current_time = datetime.datetime.now()
        for file in glob.glob("trained_model_*.spacy"):
            file_time = datetime.datetime.fromtimestamp(os.path.getctime(file))
            if (current_time - file_time).days > 7:  # Remove files older than 7 days
                os.remove(file)
                logger.info(f"Removed old training file: {file}")

    def save_trained_model(self):
        """Save the trained model."""
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"trained_models/trained_model_{timestamp}.spacy"
        self.nlp.to_disk(model_path)
        logger.info(f"Saved trained model to {model_path}")