"""
Discovery Proxy Module

Defines proxies for when signals became "known" to the market.
This is the hardest and most interesting part - we're not claiming exact discovery,
only reasonable proxies.
"""

from datetime import datetime
from typing import Dict, Optional
import pandas as pd


class DiscoveryProxy:
    """Base class for discovery proxies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        """
        Get the proxy discovery date for a signal.
        
        Returns:
            datetime or None if not available
        """
        raise NotImplementedError


# Historical discovery dates (well-researched proxies)
DISCOVERY_DATES = {
    'momentum_12_1': {
        'academic_paper': datetime(1993, 6, 1),  # Jegadeesh & Titman (1993)
        'popular_book': datetime(2000, 1, 1),    # Widespread in practitioner literature
        'blog_mentions': datetime(2005, 1, 1),   # Quant blogs popularized
    },
    'mean_reversion': {
        'academic_paper': datetime(1990, 1, 1),  # Poterba & Summers (1988), Lo & MacKinlay (1990)
        'popular_book': datetime(1996, 1, 1),    # Common in technical analysis books
        'blog_mentions': datetime(2004, 1, 1),
    },
    'volatility_breakout': {
        'academic_paper': datetime(1980, 1, 1),  # Bollinger Bands (1980s)
        'popular_book': datetime(1992, 1, 1),    # Bollinger on Bollinger Bands
        'blog_mentions': datetime(2006, 1, 1),
    },
    'ma_crossover': {
        'academic_paper': datetime(1970, 1, 1),  # Very old technical analysis
        'popular_book': datetime(1980, 1, 1),    # Widespread in TA literature
        'blog_mentions': datetime(2003, 1, 1),
    },
    'value': {
        'academic_paper': datetime(1992, 1, 1),  # Fama & French (1992)
        'popular_book': datetime(1998, 1, 1),    # Graham & Dodd style analysis
        'blog_mentions': datetime(2004, 1, 1),
    },
}


class AcademicPaperProxy(DiscoveryProxy):
    """
    Proxy: First major academic paper publication.
    
    Conservative proxy - signals known to academia first.
    """
    
    def __init__(self):
        super().__init__(
            name="Academic Paper",
            description="First major academic publication date"
        )
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        return DISCOVERY_DATES.get(signal_name, {}).get('academic_paper')


class PopularBookProxy(DiscoveryProxy):
    """
    Proxy: First appearance in popular practitioner books.
    
    Signals become "known" when they appear in widely-read books.
    """
    
    def __init__(self):
        super().__init__(
            name="Popular Book",
            description="First appearance in widely-read practitioner books"
        )
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        return DISCOVERY_DATES.get(signal_name, {}).get('popular_book')


class BlogMentionsProxy(DiscoveryProxy):
    """
    Proxy: Widespread mentions in finance blogs.
    
    More modern proxy - when signals became common knowledge in quant blogosphere.
    """
    
    def __init__(self):
        super().__init__(
            name="Blog Mentions",
            description="Widespread mentions in popular finance/quant blogs"
        )
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        return DISCOVERY_DATES.get(signal_name, {}).get('blog_mentions')


class ConservativeProxy(DiscoveryProxy):
    """
    Conservative proxy: Use the earliest date among all proxies.
    
    This assumes signals decay from the earliest possible discovery point.
    """
    
    def __init__(self):
        super().__init__(
            name="Conservative (Earliest)",
            description="Earliest date among all proxies - most conservative"
        )
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        dates = DISCOVERY_DATES.get(signal_name, {})
        if not dates:
            return None
        # Return earliest date
        return min(d for d in dates.values() if d is not None)


class AggressiveProxy(DiscoveryProxy):
    """
    Aggressive proxy: Use the latest date among all proxies.
    
    This assumes signals only start decaying when widely known (books/blogs).
    """
    
    def __init__(self):
        super().__init__(
            name="Aggressive (Latest)",
            description="Latest date among all proxies - assumes decay starts later"
        )
    
    def get_discovery_date(self, signal_name: str) -> Optional[datetime]:
        dates = DISCOVERY_DATES.get(signal_name, {})
        if not dates:
            return None
        # Return latest date
        return max(d for d in dates.values() if d is not None)


# Proxy registry
PROXY_REGISTRY = {
    'academic': AcademicPaperProxy(),
    'book': PopularBookProxy(),
    'blog': BlogMentionsProxy(),
    'conservative': ConservativeProxy(),
    'aggressive': AggressiveProxy(),
}


def get_proxy(name: str) -> DiscoveryProxy:
    """Get discovery proxy by name."""
    if name not in PROXY_REGISTRY:
        raise ValueError(f"Unknown proxy: {name}. Available: {list(PROXY_REGISTRY.keys())}")
    return PROXY_REGISTRY[name]


def list_proxies() -> Dict[str, str]:
    """List all available proxies with descriptions."""
    return {name: proxy.description for name, proxy in PROXY_REGISTRY.items()}


def get_discovery_date(signal_name: str, proxy_name: str = 'conservative') -> Optional[datetime]:
    """Convenience function to get discovery date."""
    proxy = get_proxy(proxy_name)
    return proxy.get_discovery_date(signal_name)

