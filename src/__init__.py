"""
Initialization file for the Trendify package.

This file ensures that the `src` folder is treated as a Python package and allows
modules within the package to be imported.

Author: Trendify Team
"""

from .preprocessing import (
    preprocess_business,
    preprocess_reviews,
    preprocess_users,
    preprocess_tips,
    preprocess_checkins,
)

from .models import (
    train_business_model,
    train_review_model,
    train_user_model,
    train_checkin_model,
)

from .meta_model import train_meta_model

from .visualization import (
    plot_rating_distribution,
    plot_trend_over_time,
)

__version__ = "1.0.0"

__all__ = [
    "preprocess_business",
    "preprocess_reviews",
    "preprocess_users",
    "preprocess_tips",
    "preprocess_checkins",
    "train_business_model",
    "train_review_model",
    "train_user_model",
    "train_checkin_model",
    "train_meta_model",
    "plot_rating_distribution",
    "plot_trend_over_time",
]
