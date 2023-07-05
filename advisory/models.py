from pydantic import BaseModel


class Advisory(BaseModel):
    url: str
    id: str
    title: str
    severity: str
    cveList: str
    cvsScore: str
    summary: str
    affectedProducts: str
    firstPublished: str
    details: str
    workarounds: str
    fixedSoftware: str
    exploitationPublicAnnouncements: str
    source: str
