import mechanicalsoup
import nltk
import networkx as nx
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string

# دانلود منابع مورد نیاز NLTK (فقط در اولین اجرا)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TeaSummarizer:
    def __init__(self):
        self.browser = mechanicalsoup.StatefulBrowser(user_agent='Mozilla/5.0')
        self.stop_words = set(stopwords.words('english') | set(string.punctuation))
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fetch_page_content(self, url: str) -> str:
        """Fetch and clean page content from a single URL"""
        try:
            self.browser.open(url, timeout=5)
            soup = self.browser.get_current_page()
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'head', 'iframe']):
                element.decompose()
                
            # Get text from paragraphs and clean it
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            clean_text = ' '.join(paragraphs)
            clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove extra whitespace
            return clean_text
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    def request(self) -> str:
        """Fetch content from multiple URLs concurrently"""
        urls = [
            "https://en.wikipedia.org/wiki/Tea",
            "https://www.teaclass.com/lesson_0101.html",
            "https://www.arborteas.com/what-is-tea/"
        ]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.fetch_page_content, urls))
        
        return ' '.join(results)

    def preprocess(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """Preprocess text into sentences and cleaned words"""
        sentences = nltk.sent_tokenize(text)
        
        # Advanced cleaning
        words = []
        for sent in sentences:
            # Remove punctuation and numbers
            cleaned = re.sub(r'[^\w\s]', '', sent.lower())
            cleaned = re.sub(r'\d+', '', cleaned)
            # Tokenize and remove stopwords
            tokens = [w for w in nltk.word_tokenize(cleaned) 
                     if w not in self.stop_words and len(w) > 2]
            words.append(tokens)
            
        return sentences, words

    def similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """Enhanced similarity measure using TF-IDF and Jaccard"""
        # Jaccard similarity
        set1 = set(sent1)
        set2 = set(sent2)
        if not set1 or not set2:
            return 0.0
            
        jaccard = len(set1 & set2) / len(set1 | set2)
        
        # TF-IDF cosine similarity
        tfidf_sim = 0.0
        try:
            tfidf_matrix = self.vectorizer.fit_transform([' '.join(sent1), ' '.join(sent2)])
            tfidf_sim = (tfidf_matrix * tfidf_matrix.T).A[0,1]
        except:
            pass
            
        # Combined score
        return 0.6 * jaccard + 0.4 * tfidf_sim

    def build_matrix(self, sentences: List[List[str]]) -> np.ndarray:
        """Build similarity matrix with optimizations"""
        n = len(sentences)
        matrix = np.zeros((n, n))
        
        # Precompute TF-IDF for all sentences
        sentence_texts = [' '.join(sent) for sent in sentences]
        self.vectorizer.fit(sentence_texts)
        
        for i in range(n):
            for j in range(i, n):  # Take advantage of symmetric matrix
                score = self.similarity(sentences[i], sentences[j])
                matrix[i][j] = score
                matrix[j][i] = score
                
        return matrix

    def summarize(self, text: str, n: int = 4) -> str:
        """Enhanced summarization with keyphrase extraction"""
        sentences, words = self.preprocess(text)
        
        if len(sentences) < 2:
            return text[:500] + "..." if len(text) > 500 else text
            
        matrix = self.build_matrix(words)
        graph = nx.from_numpy_array(matrix)
        scores = nx.pagerank(graph)
        
        # Get top sentences
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary_sentences = [s for _, s in ranked[:n]]
        
        # Extract keyphrases
        keyphrases = self.extract_keyphrases(words)
        
        # Build final summary
        summary = " ".join(summary_sentences)
        if keyphrases:
            summary += f"\n\nKey Topics: {', '.join(keyphrases[:5])}"
            
        return summary

    def extract_keyphrases(self, tokenized_sentences: List[List[str]]) -> List[str]:
        """Extract important keyphrases using TF-IDF and frequency"""
        # Flatten all tokens
        all_words = [word for sent in tokenized_sentences for word in sent]
        
        # Get most common meaningful words
        word_freq = Counter(all_words)
        common_words = [word for word, count in word_freq.most_common(20) 
                       if count > 1 and len(word) > 3]
        
        # Get bigrams
        bigrams = []
        for sent in tokenized_sentences:
            bigrams.extend(list(nltk.ngrams(sent, 2)))
        bigram_freq = Counter([' '.join(bg) for bg in bigrams])
        common_bigrams = [bg for bg, count in bigram_freq.most_common(10) 
                        if count > 1]
        
        return common_words + common_bigrams

# Usage
if __name__ == "__main__":
    summarizer = TeaSummarizer()
    tea_content = summarizer.request()
    summary = summarizer.summarize(tea_content, n=4)
    
    print("="*80)
    print("Tea Summary:")
    print("="*80)
    print(summary)
    print("="*80)
