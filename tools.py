from config import dense_index

def ragQueryTool(query):
    """
    Perform a RAG query on the dense index.
    """
    
    try:
        results = dense_index.search(
            namespace="chatpdf",
            query={
                "top_k": 10,
                "inputs": {
                    "text": query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }   
        )
        
        print(f"RAG query results for '{query}':", results)
        # for hit in results['result']['hits']:
        #     print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | text: {hit['fields']['chunk_text']:<50}")
        
        return results
    except Exception as e:
        print(f"Error performing RAG query: {e}")
        return None
        