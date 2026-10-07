from pydantic import BaseModel, Field


class AgriculturalResponse(BaseModel):

    disease: str = Field(
        description="The name of the plant disease"
    )

    symptoms: list[str] = Field(
        description="Symptoms of the disease"
    )

    management: list[str] = Field(
        description="General management recommendations"
    )

    prevention: list[str] = Field(
        description="Prevention recommendations"
    )