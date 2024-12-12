import spacy

# Load spaCy's model for semantic similarity
nlp = spacy.load("en_core_web_md")

# Groundwater-related phrases
groundwater_phrases = [
    "Groundwater Management",
    "Central Ground Water Authority (CGWA)",
    "Ministry of Jal Shakti",
    "Hydrogeological Scenario",
    "Water Level Data",
    "Water Quality",
    "Groundwater Resources",
    "Comprehensive Report",
    "Categorization of Areas",
    "Groundwater Assessment",
    "ground water",
    "aquifier",
    "wells",
    "subsurface water",
    "hydrology",
    "hydrogeology",
    "hydrogeological"
    "water level",
    "Aquifer", "Water table", "Recharge", "Groundwater flow", "Subsurface water",
    "Well drilling", "Borehole", "Percolation", "Porosity", "Permeability",
    "Artesian well", "Confined aquifer", "Unconfined aquifer", "Hydraulic gradient",
    "Hydrogeology", "Groundwater contamination", "Groundwater management",
    "Saltwater intrusion", "Aquifer depletion", "Drawdown", "Seepage",
    "Karst system", "Groundwater recharge zone", "Over-extraction", "Springs",
    "Surface water", "Freshwater", "Watershed", "Water cycle", "Precipitation",
    "Evaporation", "Condensation", "Infiltration", "Runoff", "Water conservation",
    "Water quality", "Water treatment", "Desalination", "Water purification",
    "Greywater", "Rainwater harvesting", "Irrigation", "Flood management",
    "Drought resilience", "Hydrology", "Hydroelectricity", "Drinking water",
    "Sanitation", "Water rights", "Water allocation", "Aquatic ecosystems",
    "Nitrate contamination", "Pesticide runoff", "Industrial discharge",
    "Heavy metals", "Eutrophication", "Leachate", "Oil spill", "Microplastics",
    "Contaminant plume", "Waterborne pathogens", "Bioaccumulation",
    "Sewage overflow", "Point-source pollution", "Non-point source pollution",
    "Climate change", "Drought", "Flood", "Monsoon", "Glacier melt",
    "Sea level rise", "Water scarcity", "Riparian zones", "Wetlands",
    "Desertification", "Ecosystem services", "Soil erosion",
    "Sustainable water management", "River basin management",
    "Integrated water resources management (IWRM)",
    "Water security", "Water governance", "Transboundary water",
    "Water infrastructure", "Irrigation systems", "Urban water supply",
    "Agricultural water use", "Industrial water use", "Water pricing",
    "Sustainable development goals (SDG 6: Clean water and sanitation)",
    "Water equity", "Public health and water", "Waterborne diseases",
    "Hydraulic conductivity", "Groundwater modeling", "Monitoring wells",
    "Water sampling", "Hydrometer", "Geothermal energy", "Water sensors",
    "Digital hydrology", "Aquifer storage and recovery (ASR)", "Pumping test",
    "Groundwater", "Water Level", "Hydrogeology", "Aquifer", "Water Quality", "Groundwater Reports", "NOC", "Groundwater Management", "Aquifer Levels", "Seasonal Fluctuations", "Water Table Depth", "Pre-monsoon Data", "Post-monsoon Data", "Permeability", "Porosity", "Recharge Zones", "Geological Formations", "Transmissivity", "Salinity", "pH Levels", "Contamination", "Arsenic", "Fluoride", "Nitrate", "Pollution Sources", "Potable Water Standards", "Ground Water Yearbooks", "Aquifer Mapping", "State Reports", "Regional Reports", "Rainwater Harvesting", "Aquifer Recharge", "Sustainable Policies", "Conservation", "Application Process", "Industrial Regulations", "Domestic Use", "Agricultural Use", "Legal Framework", "Central Guidelines", "State Guidelines",
    "Groundwater Resource Assessment", "Categorization of Areas", "Water Resource Reports", "Groundwater Flow",
    "Drawdown", "Over-exploited Areas", "Recharge Techniques", "Water Quality Monitoring", "Groundwater Pollution",
    "Recharge Wells", "Water Budgeting", "Rainfall Data", "Well Drilling Guidelines", "Sustainable Extraction",
    "Groundwater Conservation Plans", "Water Availability", "Hydrological Cycle", "Groundwater Monitoring",
    "Water Sampling", "Groundwater Data Analysis", "Aquifer Storage", "Groundwater Recharge Zones",
    "Groundwater Depletion", "Aquifer Characteristics", "Groundwater Sustainability", "Water Scarcity Solutions",
    "Water Management Practices", "Groundwater Extraction Policies", "Groundwater Use Optimization", "Water Allocation", "Hydrograph Analysis", "Groundwater Legislation", "Water Demand Assessment", "Groundwater Resource Planning", "Groundwater Development", "Water Quality Standards", "Groundwater Recharge Enhancement", "Water Use Efficiency", "Community Participation", "Aquifer Recharge Mechanisms", "Water Table Mapping", "Groundwater Overuse", "Groundwater Contamination Sources", "Hydrogeological Data", "Environmental Impact", "Groundwater Training Programs"
]

# Function to check relevance
def is_related_to_groundwater(question, threshold=0.5):
    question_doc = nlp(question)
    for phrase in groundwater_phrases:
        phrase_doc = nlp(phrase)
        if question_doc.similarity(phrase_doc) > threshold:
            return True
    return False
