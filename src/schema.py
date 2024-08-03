from pydantic import BaseModel

class Medical(BaseModel):
    id: str
    subject: str
    message: str
    answer: str
    focus: str
    type: str

    def __repr__(self):
        return f"Medical(subject={self.subject}, message={self.message}, focus={self.focus})"

    def __str__(self):
        return self.__repr__()
    