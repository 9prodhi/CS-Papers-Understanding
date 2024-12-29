# Knowledge-Aware Query Expansion with Large Language Models for Textual and Relational Retrieval

In the context of Retrieval-Augmented Generation, retrieving highly relevant documents requires not only understanding the text but also the relationships between entities within the corpus. Consider a scenario where you are searching for complex information, such as academic papers or products, where the query involves both textual content and relational aspects. For example, a query like "a paper about machine learning that cites another paper by Dr. Smith and was published in 2020" includes textual components (machine learning, 2020) and relational components (cites another paper, authored by Dr. Smith).

This paper (https://arxiv.org/pdf/2410.13765) focuses on addressing this challenge by enhancing text retrieval using *query expansion*, a technique that incorporates both textual and relational information. The system works with two primary components:

1. **Textual Documents (D)**: A collection of documents, each describing entities like academic papers (e.g., titles, abstracts).
2. **Knowledge Graph (G)**: A network of entities (e.g., papers, authors, venues) connected by relationships (e.g., "cites", "authored by", "published at").

In this setup:
- Each document \( d_i \) describes an entity (like a paper).  
- Each entity \( v_i \) has a corresponding node in the knowledge graph.  
- Relations (R) connect these nodes (for example, "paper A cites paper B").  

#### The Problem: Semi-Structured Retrieval
You have a **query** \( q \) that requires both textual and relational information. The goal is to find documents \( A \subseteq D \) that match the text and the relations specified in \( q \).

#### The Solution: Query Expansion
Instead of searching with the query as it is, we expand it using extra information from the knowledge base. This makes the query more detailed and aligned with the documents.  

Here’s how it works:
1. **Start with a Query** – For example, **"Find papers by Dr. Smith on deep learning from 2020."**  
2. **Extract Related Information** – We use the existing documents and knowledge graph to gather more context.  
3. **Expand the Query** – The system generates additional terms and relationships to add to the original query.  

#### Formula Breakdown:
- \( Q_e = f(q, D, G) \)  
   - \( f \) is a function that looks at the query, documents, and the knowledge graph to generate expansion terms \( Q_e \).  
- \( q' = \text{Concat}(q, Q_e) \)  
   - The expanded query \( q' \) combines the original query \( q \) and the new expansion \( Q_e \).  

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
