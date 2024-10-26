'''
arxiv: 2.1.3
scholarly: 1.7.1
'''


import arxiv
from scholarly import scholarly
from tqdm

def get_affiliation_by_name(name):
    try:
        # Search for the author by name
        search_query = scholarly.search_author(name)
        author = next(search_query)  # Get the first result

        # Get the affiliation
        if "affiliation" in author.keys():
            affiliation = author["affiliation"]
        else:
            affiliation = None
        return {
            "name": author["name"] if "name" in author.keys() else name,
            "affiliation": affiliation
        }
    except StopIteration:
        return {
            "name": name,
            "affiliation": "Not found"
        }
    except Exception as e:
        return {
            "name": name,
            "affiliation": f"Error: {str(e)}"
        }

def get_last_authors_and_affiliations(titles):
    authors_info = []
    
    for title in tqdm(titles):
        # Search for the paper by title
        search = arxiv.Search(query=title, max_results=1)
        result = next(search.results(), None)
        
        if result:
            last_author = str(result.authors[-1])  # Get the last author
            affiliation = get_affiliation_by_name(last_author)
            authors_info.append({
                "title": title,
                "last_author": last_author,
                "affiliation": affiliation,
            })
        else:
            authors_info.append({
                "title": title,
                "last_author": "Not found",
                "affiliation": "Not found",
            })

    return authors_info

titles = [
    "Pix2map: Cross-Modal Retrieval for Inferring Street Maps from Images",
    "Audio-Visual Grouping Network for Sound Localization from Mixtures",
    "Learning Semantic Relationship Among Instances for Image-Text Matching",
    "Identity-Preserving Talking Face Generation with Landmark and Appearance Priors",
    "ImageBind: One Embedding Space to Bind them All",
    "Learning to Dub Movies via Hierarchical Prosody Models",
    "OmniMAE: Single Model Masked Pretraining on Images and Videos",
    "CNVid-3.5M: Build, Filter, and Pre-Train the Large-Scale Public Chinese Video-Text Dataset",
    "Egocentric Audio-Visual Object Localization",
    "Learning Visual Representations via Language-Guided Sampling",
    "Unite and Conquer: Plug and Play Multi-Modal Synthesis using Diffusion Models",
    "iQuery: Instruments As Queries for Audio-Visual Sound Separation",
    "Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-Identification",
    "PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection",
    "Prompt, Generate, then Cache: Cascade of Foundation Models Makes Strong Few-Shot Learners",
    "Non-Contrastive Learning Meets Language-Image Pre-Training",
    "Highly Confident Local Structure based Consensus Graph Learning for Incomplete Multi-View Clustering",
    "Vision Transformers are Parameter-Efficient Audio-Visual Learners",
    "Teaching Structured Vision and Language Concepts to Vision and Language Models",
    "Data-Free Sketch-based Image Retrieval",
    "Align and Attend: Multimodal Summarization with Dual Contrastive Losses",
    "Efficient Multimodal Fusion via Interactive Prompting",
    "Multimodal Prompting with Missing Modalities for Visual Recognition",
    "Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce",
    "What Happened 3 Seconds Ago? Inferring the Past with Thermal Imaging",
    "MMANet: Margin-Aware Distillation and Modality-Aware Regularization for Incomplete Multimodal Learning",
    "Multi-Modal Learning with Missing Modality via Shared-Specific Feature Modelling",
    "The ObjectFolder Benchmark: Multisensory Learning with Neural and Real Objects",
    "Position-Guided Text Prompt for Vision-Language Pre-Training",
    "Conditional Generation of Audio from Video via Foley Analogies",
    "OSAN: A One-Stage Alignment Network to Unify Multimodal Alignment and Unsupervised Domain Adaptation",
    "Self-Supervised Video Forensics by Audio-Visual Anomaly Detection",
    "ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding",
    "AVFormer: Injecting Vision into Frozen Speech Models for Zero-Shot AV-ASR",
    "Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring",
    "SceneTrilogy: On Human Scene-Sketch and its Complementarity with Photo and Text",
    "Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification",
    "EXIF as Language: Learning Cross-Modal Associations between Images and Camera Metadata",
    "Revisiting Multimodal Representation in Contrastive Learning: From Patch and Token Embeddings to Finite Discrete Tokens",
    "RONO: Robust Discriminative Learning with Noisy Labels for 2D-3D Cross-Modal Retrieval",
    "CASP-Net: Rethinking Video Saliency Prediction from an Audio-Visual Consistency Perceptual Perspective",
    "Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning",
    "ReVISE: Self-Supervised Speech Resynthesis with Visual Input for Universal and Generalized Speech Regeneration",
    "Look, Radiate, and Learn: Self-Supervised Localisation via Radio-Visual Correspondence",
    "Learning Emotion Representations from Verbal and Nonverbal Communication",
    "Enhanced Multimodal Representation Learning with Cross-Modal KD",
    "MELTR: Meta Loss Transformer for Learning to Fine-Tune Video Foundation Models",
    "Multilateral Semantic Relations Modeling for Image Text Retrieval",
    "GeoVLN: Learning Geometry-Enhanced Visual Representation with Slot Attention for Vision-and-Language Navigation",
    "Noisy Correspondence Learning with Meta Similarity Correction",
    "Improving Cross-Modal Retrieval with Set of Diverse Embeddings",
    "Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment",
    "MaPLe: Multi-Modal Prompt Learning",
    "Fine-Grained Image-Text Matching by Cross-Modal Hard Aligning Network",
    "Towards Modality-Agnostic Person Re-Identification with Descriptive Query",
    "Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos",
    "FashionSAP: Symbols and Attributes Prompt for Fine-Grained Fashion Vision-Language Pre-Training",
    "MAP: Multimodal Uncertainty-Aware Vision-Language Pre-Training Model",
    "Egocentric Auditory Attention Localization in Conversations",
    "Improving Zero-Shot Generalization and Robustness of Multi-Modal Models",
    "Understanding and Constructing Latent Modality Structures in Multi-Modal Representation Learning",
    "Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles",
    "GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering",
    "BiCro: Noisy Correspondence Rectification for Multi-Modality Data via Bi-Directional Cross-Modal Similarity Consistency",
    "DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training",
    "Referring Image Matting",
    "Leveraging per Image-Token Consistency for Vision-Language Pre-Training",
    "Seeing what You Miss: Vision-Language Pre-Training with Semantic Completion Learning",
    "Sample-Level Multi-View Graph Clustering",
    "SmallCap: Lightweight Image Captioning Prompted with Retrieval Augmentation",
    "On the Effects of Self-Supervision and Contrastive Alignment in Deep Multi-View Clustering",
    "SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model",
    "Novel-View Acoustic Synthesis",
    "MAGVLT: Masked Generative Vision-and-Language Transformer",
    "Reproducible Scaling Laws for Contrastive Language-Image Learning",
    "PMR: Prototypical Modal Rebalance for Multimodal Learning",
    "Language-Guided Music Recommendation for Video via Prompt Analogies",
    "RA-CLIP: Retrieval Augmented Contrastive Language-Image Pre-Training",
    "MMG-Ego4D: Multimodal Generalization in Egocentric Action Recognition",
    "Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning",
    "PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment",
    "Masked Autoencoding does not Help Natural Language Supervision at Scale",
    "CLIPPO: Image-and-Language Understanding from Pixels Only",
    "Chat2Map: Efficient Scene Mapping from Multi-Ego Conversations",
    "Critical Learning Periods for Multisensory Integration in Deep Networks",
    "CLIPPING: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval",
    "NUWA-LIP: Language-Guided Image Inpainting with Defect-Free VQGAN",
    "WINNER: Weakly-Supervised hIerarchical decompositioN and aligNment for Spatio-tEmporal Video gRounding",
    "Multivariate, Multi-Frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation"
]


if __name__ == "__main__":
	authors = get_last_authors_and_affiliations(titles)
	print(authors)