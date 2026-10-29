from fastapi import APIRouter

from app.controllers.config_controller import ConfigController

router = APIRouter(tags=["config"])


@router.get("/config")
async def config():
    return ConfigController.config()
