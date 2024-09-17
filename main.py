# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tier1_analyzer import AdvancedDrugAnalyzer

app = FastAPI()

analyzer = AdvancedDrugAnalyzer()

class Prescription(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    drug: str
    initial_dosage: str
    initial_frequency: str
    dosage_route: str
    conditional_instructions: str | None
    dosage_changes: list
    duration: str | None
    confidence: float
    entities: dict

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_prescription(prescription: Prescription):
    try:
        result = analyzer.analyze(prescription.text)
        return AnalysisResult(
            drug=result.drug,
            initial_dosage=result.initial_dosage,
            initial_frequency=result.initial_frequency,
            dosage_route=result.dosage_route,
            conditional_instructions=result.conditional_instructions,
            dosage_changes=result.dosage_changes,
            duration=result.duration,
            confidence=result.confidence,
            entities=result.entities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)