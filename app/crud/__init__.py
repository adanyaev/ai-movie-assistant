from app.models.user import User as UserModel
from app.models.message import Message as MessageModel

from app.schemas.user import User as UserSchema
from app.schemas.message import Message as MessageSchema

from .user import CRUDUser
from .message import CRUDMessage

user = CRUDUser(UserModel, UserSchema)
message = CRUDMessage(MessageModel, MessageSchema)
