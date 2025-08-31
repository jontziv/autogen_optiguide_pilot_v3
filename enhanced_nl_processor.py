#!/usr/bin/env python3
"""
Enhanced Natural Language Processing for OptiGuide
=================================================
Handles complex what-if scenarios with multiple SKU-specific changes,
service level adjustments, and lead time modifications.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedChange:
    """Represents a parsed change from natural language"""
    parameter: str
    value: float
    direction: str  # 'increase', 'decrease', 'set'
    target: Optional[str] = None  # SKU ID, location, or None for global
    confidence: float = 1.0

class EnhancedNLProcessor:
    """Enhanced natural language processing for complex what-if scenarios"""
    
    def __init__(self):
        # Improved patterns for better extraction
        self.parameter_patterns = {
            'demand': [
                r'(?:avg_daily_)?demand.*?(?:for\s+)?(sku[-\s]*[0-9a-z]+)?.*?(?:increases?|rises?|goes?\s+up).*?(\d+(?:\.\d+)?)\s*%',
                r'(?:avg_daily_)?demand.*?(?:for\s+)?(sku[-\s]*[0-9a-z]+)?.*?(?:decreases?|drops?|falls?|goes?\s+down).*?(\d+(?:\.\d+)?)\s*%',
                r'(sku[-\s]*[0-9a-z]+).*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%',
                r'(sku[-\s]*[0-9a-z]+).*?(?:decreases?|drops?|falls?).*?(\d+(?:\.\d+)?)\s*%',
            ],
            'service_level': [
                r'service\s+level.*?(?:increases?|rises?|target|set).*?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s+service\s+level',
            ],
            'lead_time': [
                r'lead\s+time.*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%',
                r'lead\s+time.*?(?:decreases?|drops?).*?(\d+(?:\.\d+)?)\s*%',
            ]
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse a natural language what-if question into structured scenario"""
        logger.info(f"Parsing question: {question}")
        
        changes = self._extract_all_changes(question)
        scenario = self._build_scenario_from_changes(changes, question)
        
        logger.info(f"Extracted {len(changes)} changes")
        for change in changes:
            logger.info(f"  - {change.parameter}: {change.value}% {change.direction} for {change.target or 'all'}")
        
        return scenario
    
    def _extract_all_changes(self, text: str) -> List[ParsedChange]:
        """Extract all parameter changes from the text"""
        text_lower = text.lower()
        changes = []
        
        # Extract demand changes with SKU specificity
        changes.extend(self._extract_demand_changes(text_lower))
        
        # Extract service level changes
        changes.extend(self._extract_service_level_changes(text_lower))
        
        # Extract lead time changes
        changes.extend(self._extract_lead_time_changes(text_lower))
        
        return changes
    
    def _extract_demand_changes(self, text: str) -> List[ParsedChange]:
        """Extract demand-related changes"""
        changes = []
        
        # Pattern 1: "avg_daily_demand increases by 90% for SKU-0001"
        pattern1 = r'(?:avg_daily_)?demand.*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%.*?(?:for\s+)?(sku[-\s]*[0-9a-z]+)'
        for match in re.finditer(pattern1, text):
            percentage = float(match.group(1))
            sku = self._normalize_sku(match.group(2))
            changes.append(ParsedChange('demand', percentage, 'increase', sku))
        
        # Pattern 2: "SKU-0001 increases by 90%"
        pattern2 = r'(sku[-\s]*[0-9a-z]+).*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(pattern2, text):
            sku = self._normalize_sku(match.group(1))
            percentage = float(match.group(2))
            changes.append(ParsedChange('demand', percentage, 'increase', sku))
        
        # Pattern 3: "demand for SKU-0003 decreases by 30%"
        pattern3 = r'(?:avg_daily_)?demand.*?(?:for\s+)?(sku[-\s]*[0-9a-z]+).*?(?:decreases?|drops?|falls?).*?(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(pattern3, text):
            sku = self._normalize_sku(match.group(1))
            percentage = float(match.group(2))
            changes.append(ParsedChange('demand', percentage, 'decrease', sku))
        
        # Pattern 4: "SKU-0003 decreases by 30%"  
        pattern4 = r'(sku[-\s]*[0-9a-z]+).*?(?:decreases?|drops?|falls?).*?(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(pattern4, text):
            sku = self._normalize_sku(match.group(1))
            percentage = float(match.group(2))
            changes.append(ParsedChange('demand', percentage, 'decrease', sku))
        
        # Pattern 5: Global demand changes
        pattern5 = r'(?:avg_daily_)?demand.*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%'
        if not any('sku' in change.target for change in changes if change.target):
            for match in re.finditer(pattern5, text):
                percentage = float(match.group(1))
                if not any(c.parameter == 'demand' and c.target for c in changes):
                    changes.append(ParsedChange('demand', percentage, 'increase', None))
        
        return changes
    
    def _extract_service_level_changes(self, text: str) -> List[ParsedChange]:
        """Extract service level changes"""
        changes = []
        
        # Pattern: "service level to 98%" or "98% service level"
        patterns = [
            r'service\s+level.*?(?:to|target|set).*?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s+service\s+level'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                percentage = float(match.group(1))
                if percentage >= 80 and percentage <= 99.9:  # Reasonable service level range
                    changes.append(ParsedChange('service_level', percentage, 'set', None))
        
        return changes
    
    def _extract_lead_time_changes(self, text: str) -> List[ParsedChange]:
        """Extract lead time changes"""
        changes = []
        
        patterns = [
            r'lead\s+time.*?(?:increases?|rises?).*?(\d+(?:\.\d+)?)\s*%',
            r'lead\s+time.*?(?:decreases?|drops?).*?(\d+(?:\.\d+)?)\s*%'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                percentage = float(match.group(1))
                direction = 'increase' if any(word in match.group(0) for word in ['increase', 'rise']) else 'decrease'
                changes.append(ParsedChange('lead_time', percentage, direction, None))
        
        return changes
    
    def _normalize_sku(self, sku_text: str) -> str:
        """Normalize SKU text to standard format"""
        sku_text = sku_text.strip().upper()
        if not sku_text.startswith('SKU'):
            # Handle cases like "0001" -> "SKU-0001"
            if sku_text.isdigit():
                sku_text = f"SKU-{sku_text.zfill(4)}"
            else:
                sku_text = f"SKU-{sku_text}"
        elif 'SKU' in sku_text and '-' not in sku_text:
            # Handle cases like "SKU0001" -> "SKU-0001" 
            sku_text = sku_text.replace('SKU', 'SKU-')
        return sku_text
    
    def _build_scenario_from_changes(self, changes: List[ParsedChange], question: str) -> Dict[str, Any]:
        """Build scenario dictionary from parsed changes"""
        scenario = {
            "changes": {},
            "run_options": {
                "scenario_name": self._generate_scenario_name(question),
                "question_type": self._classify_question_type(changes)
            }
        }
        
        # Process demand changes
        demand_changes = [c for c in changes if c.parameter == 'demand']
        if demand_changes:
            scenario["changes"]["demand_multiplier"] = self._build_demand_multiplier(demand_changes)
        
        # Process service level changes
        service_level_changes = [c for c in changes if c.parameter == 'service_level']
        if service_level_changes:
            # Use the last/highest service level found
            target_sl = max(c.value for c in service_level_changes) / 100.0
            scenario["changes"]["service_level_target"] = {"default": target_sl}
        
        # Process lead time changes
        lead_time_changes = [c for c in changes if c.parameter == 'lead_time']
        if lead_time_changes:
            # Use the last lead time change found
            change = lead_time_changes[-1]
            multiplier = 1.0 + (change.value / 100.0 * (1 if change.direction == 'increase' else -1))
            scenario["changes"]["lead_time_days"] = {"multiplier": multiplier}
        
        return scenario
    
    def _build_demand_multiplier(self, demand_changes: List[ParsedChange]) -> Dict[str, Any]:
        """Build demand multiplier structure from demand changes"""
        global_multiplier = 1.0
        overrides = {}
        
        for change in demand_changes:
            multiplier = 1.0 + (change.value / 100.0 * (1 if change.direction == 'increase' else -1))
            
            if change.target:
                # SKU-specific change
                overrides[change.target] = multiplier
            else:
                # Global change
                global_multiplier = multiplier
        
        return {
            "default": global_multiplier,
            "overrides": overrides
        }
    
    def _classify_question_type(self, changes: List[ParsedChange]) -> str:
        """Classify the type of what-if question"""
        if any(c.parameter == 'service_level' for c in changes):
            return "service_level_analysis"
        elif any(c.parameter == 'lead_time' for c in changes):
            return "lead_time_analysis"
        elif any(c.parameter == 'demand' for c in changes):
            return "demand_analysis" 
        else:
            return "general_analysis"
    
    def _generate_scenario_name(self, question: str) -> str:
        """Generate a descriptive scenario name"""
        words = question.split()[:6]
        name = "_".join(word.lower().strip("?.,!") for word in words if len(word) > 2)
        return name[:50]  # Limit length

# Test the enhanced processor
if __name__ == "__main__":
    processor = EnhancedNLProcessor()
    
    test_questions = [
        "what if avg_daily_demand increases by 90% for SKU-0001 and avg_daily_demand decreases by 30% for SKU-0003?",
        "what if demand for SKU-0001 goes up 50% and service level target is 98%?",
        "what if lead times increase by 20% and we target 95% service level?",
        "what if SKU-0002 increases 75% and SKU-0003 drops by 25%?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        scenario = processor.parse_question(question)
        print(f"Scenario: {json.dumps(scenario, indent=2)}")