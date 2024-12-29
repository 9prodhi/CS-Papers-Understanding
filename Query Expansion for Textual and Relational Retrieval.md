# Knowledge-Aware Query Expansion with Large Language Models for Textual and Relational Retrieval

In the context of Retrieval-Augmented Generation, retrieving highly relevant documents requires not only understanding the text but also the relationships between entities within the corpus. Consider a scenario where you are searching for complex information, such as academic papers or products, where the query involves both textual content and relational aspects. For example, a query like "a paper about machine learning that cites another paper by Dr. Smith and was published in 2020" includes textual components (machine learning, 2020) and relational components (cites another paper, authored by Dr. Smith).

This paper (https://arxiv.org/pdf/2410.13765) focuses on addressing this challenge by enhancing text retrieval using *query expansion*, a technique that incorporates both textual and relational information. The system works with two primary components:

1. **Textual Documents (D)**: A collection of documents, each describing entities like academic papers (e.g., titles, abstracts).
2. **Knowledge Graph (G)**: A network of entities (e.g., papers, authors, venues) connected by relationships (e.g., "cites", "authored by", "published at").

In this setup:
- Each document $d_i$ describes an entity (like a paper).  
- Each entity $v_i$ has a corresponding node in the knowledge graph.  
- Relations $(R)$ connect these nodes (for example, "paper A cites paper B").  

#### The Problem: Semi-Structured Retrieval
You have a **query** $q$ that requires both textual and relational information. The goal is to find documents $A \subseteq D$ that match the text and the relations specified in $q$.

#### The Solution: Query Expansion
Instead of searching with the query as it is, we expand it using extra information from the knowledge base. This makes the query more detailed and aligned with the documents.  

Here’s how it works:
1. **Start with a Query** – For example, **"Find papers by Dr. Smith on deep learning from 2020."**  
2. **Extract Related Information** – Use the existing documents and knowledge graph to gather more context.  
3. **Expand the Query** – The system generates additional terms and relationships to add to the original query.  

#### Formula Breakdown:
$$ Q_e = f(q, D, G) $$  
- $f$ is a function that looks at the query, documents, and the knowledge graph to generate expansion terms $Q_e$.  

$$ q' = \text{Concat}(q, Q_e) $$  
- The expanded query $q'$ combines the original query $q$ and the new expansion $Q_e$.  

This expanded query is then used to retrieve documents that better match the original intent, improving search accuracy by incorporating both text and structured relationships.  

#### Example in Action:
Imagine you’re searching for papers. Each paper is represented in two ways:  
1. **Textual Document (D):** Contains abstract, title, etc.  
2. **Graph Node (G):** Connected to authors, citations, and topics.  

If your query asks for papers **authored by a specific person and citing a certain paper,** the system will:  
- Look for textual matches.  
- Find connections in the graph (such as citations or co-authorship).  
- Expand the query to ensure documents retrieved are highly relevant.

This process helps bridge the gap between what you’re asking and how the data is organized, ensuring more accurate and comprehensive search results.

#### KG creation sample code:

```bash
conda create -n connected-papers python=3.12
conda activate connected-papers

pip install requests
pip install igraph fake_useragent
pip install ogb
```

```python

import torch
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import pandas as pd
import igraph as ig
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivKG:
    def __init__(self):
        self.graph = ig.Graph(directed=True)
        self.vertex_index = {}  # id to vertex index mapping
        
        # ArXiv subject area categories
        self.subject_areas = {
            0: "Computer Science - Theory",
            1: "Physics - Mesoscopic Systems and Quantum Hall Effect",
            2: "Physics - Strongly Correlated Electrons",
            3: "Computer Science - Mathematical Software",
            4: "Computer Science - Social and Information Networks",
            5: "Computer Science - Computers and Society",
            6: "Computer Science - Cryptography and Security",
            7: "Computer Science - Data Structures and Algorithms",
            8: "Computer Science - Databases",
            9: "Physics - Statistical Mechanics",
            # ... Add more as needed
        }
        
    def build_from_ogb(self):
        """Build knowledge graph from ogbn-arxiv dataset."""
        logger.info("Loading ogbn-arxiv dataset...")
        dataset = NodePropPredDataset(name='ogbn-arxiv')
        graph, labels = dataset[0]  # Access the first graph
        split_idx = dataset.get_idx_split()
        
        # Get node features and edge indices
        node_feat = graph['node_feat']
        edge_index = graph['edge_index']
        node_year = graph['node_year'].squeeze()
        node_labels = labels.squeeze()
        
        # Process nodes
        logger.info("Processing nodes...")
        num_nodes = node_feat.shape[0]
        for node_id in tqdm(range(num_nodes)):
            # Add paper vertex
            self.graph.add_vertices(1)
            idx = self.graph.vcount() - 1
            self.vertex_index[node_id] = idx
            
            # Set vertex attributes
            self.graph.vs[idx]['entity_id'] = str(node_id)
            self.graph.vs[idx]['type'] = 'paper'
            self.graph.vs[idx]['year'] = int(node_year[node_id])
            self.graph.vs[idx]['subject'] = self.subject_areas.get(
                int(node_labels[node_id]), 
                f"Unknown Subject {int(node_labels[node_id])}"
            )
            self.graph.vs[idx]['embedding'] = node_feat[node_id].tolist()
            
            # Add split information (train/val/test)
            for split_name, split_indices in split_idx.items():
                if node_id in split_indices:
                    self.graph.vs[idx]['split'] = split_name
                    break
        
        # Process edges (citations)
        logger.info("Processing edges...")
        for i in tqdm(range(edge_index.shape[1])):
            source_id = int(edge_index[0, i])
            target_id = int(edge_index[1, i])
            
            # Add citation edge
            source_idx = self.vertex_index[source_id]
            target_idx = self.vertex_index[target_id]
            
            self.graph.add_edges([(source_idx, target_idx)])
            edge_idx = self.graph.get_eid(source_idx, target_idx)
            self.graph.es[edge_idx]['type'] = 'cites'
        
        logger.info(f"Built graph with {self.graph.vcount()} vertices and {self.graph.ecount()} edges")
    
    def get_paper_neighbors(self, paper_id: str, hops: int = 2, 
                          filter_year: int = None, filter_subject: str = None) -> Dict[str, List[Dict]]:
        """Get all paper neighbors within h hops with optional filtering."""
        if paper_id not in self.vertex_index:
            return {'papers': []}
            
        start_idx = self.vertex_index[paper_id]
        
        # Get vertices within h hops
        neighbors = self.graph.neighborhood(
            vertices=start_idx,
            order=hops,
            mode='all'
        )
        
        # Remove start vertex
        if start_idx in neighbors:
            neighbors.remove(start_idx)
            
        results = {'papers': []}
        
        # Collect neighbor information
        for idx in neighbors:
            vertex = self.graph.vs[idx]
            
            # Apply filters if specified
            if filter_year and vertex['year'] < filter_year:
                continue
            if filter_subject and vertex['subject'] != filter_subject:
                continue
            
            paper_data = {
                'id': vertex['entity_id'],
                'year': vertex['year'],
                'subject': vertex['subject'],
                'relation_type': self._get_relation_type(start_idx, idx),
                'split': vertex.get('split', 'unknown')
            }
            
            results['papers'].append(paper_data)
        
        return results
    
    def _get_relation_type(self, source_idx: int, target_idx: int) -> str:
        """Get the type of relation between two vertices."""
        try:
            edge = self.graph.get_eid(source_idx, target_idx)
            return self.graph.es[edge]['type']
        except:
            try:
                edge = self.graph.get_eid(target_idx, source_idx)
                return self.graph.es[edge]['type'] + '_by'
            except:
                return 'indirect'
    
    # def get_paper_embedding(self, paper_id: str) -> np.ndarray:
    #     """Get the embedding vector for a paper."""
    #     if paper_id not in self.vertex_index:
    #         return None
    #     return np.array(self.graph.vs[self.vertex_index[paper_id]]['embedding'])
    
    def get_citation_subgraph(self, paper_id: str, depth: int = 1) -> Dict:
        """Get citation network around a paper."""
        if paper_id not in self.vertex_index:
            return None
            
        start_idx = self.vertex_index[paper_id]
        
        # Get subgraph of citations
        neighbors = self.graph.neighborhood(
            vertices=start_idx,
            order=depth,
            mode='all'
        )
        
        subgraph = self.graph.subgraph(neighbors)
        
        # Convert to networkx-like dictionary format
        nodes = []
        for vertex in subgraph.vs:
            nodes.append({
                'id': vertex['entity_id'],
                'year': vertex['year'],
                'subject': vertex['subject']
            })
            
        edges = []
        for edge in subgraph.es:
            edges.append({
                'source': subgraph.vs[edge.source]['entity_id'],
                'target': subgraph.vs[edge.target]['entity_id'],
                'type': edge['type']
            })
            
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def save_graph(self, filename: str):
        """Save the graph to file."""
        self.graph.save(filename)
    
    def load_graph(self, filename: str):
        """Load the graph from file."""
        self.graph = ig.Graph.Read(filename)
        # Rebuild index
        self.vertex_index = {
            self.graph.vs[idx]['entity_id']: idx 
            for idx in range(self.graph.vcount())
        }

def main():
    # Create and build knowledge graph
    kg = ArxivKG()
    kg.build_from_ogb()
    
    # Save graph
    kg.save_graph("arxiv_kg.graphml")
    
    # Example usage
    paper_id = "0"  # first paper
    
    # Get paper neighbors
    neighbors = kg.get_paper_neighbors(
        paper_id, 
        hops=2,
        filter_year=2015  # optional: only papers after 2015
    )
    print(f"\nNeighbors of paper {paper_id}:")
    print(json.dumps(neighbors, indent=2))
    
    # Get citation subgraph
    subgraph = kg.get_citation_subgraph(paper_id)
    print(f"\nCitation subgraph size: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    
    # Get paper embedding
    # embedding = kg.get_paper_embedding(paper_id)
    # print(f"\nPaper embedding shape: {embedding.shape}")

if __name__ == "__main__":
    main()

```
