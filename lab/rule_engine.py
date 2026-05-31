import pandas as pd

"""
Motor de reglas dinámico basado en régimen de mercado + calidad fundamental.

El sistema combina:
1. Régimen de mercado (VIX-based, del regime_detector.py)
2. Señales técnicas (del indicators.py)
3. Filtro de calidad fundamental (integración con proyecto de inversión)

La señal final y los indicadores activos cambian según el régimen del día.
"""

RULE_SETS = {
    "CALM": {
        # En mercado tranquilo, EMA200 sola pierde. Solo operar con 
        # calidad fundamental alta + EMA200 como filtro de entrada.
        "requires_fundamental": True,
        "min_stars": 13,          # del proyecto de inversión
        "indicators": {
            "primary":   [("EMA200_disc", 15.0)],   # único indicador válido
            "secondary": [],                          # sin confirmadores
        },
        "min_signals": 1,
        "rationale": "CALM sin calidad = cuchillo. CALM con calidad = oportunidad táctica."
    },
    
    "SLOW_BEAR": {
        # Mercado bajando lento: osciladores + EMA200 funcionan bien combinados.
        # Calidad fundamental recomendada pero no obligatoria para ETFs.
        "requires_fundamental": False,
        "min_stars": 0,
        "indicators": {
            "primary":   [("EMA200_disc", 12.0)],
            "secondary": [("BB_pctB", 0.2), ("Stoch_K", 20.0)],
        },
        "min_signals": 2,   # primary + al menos 1 secondary
        "rationale": "Bear lento: combos de osciladores maximizan edge (+6.85pp)."
    },

    "FAST_CRASH": {
        # Crash rápido: solo EMA200 estructural. Osciladores se desactivan.
        # Calidad fundamental obligatoria (empresas débiles no rebotan).
        "requires_fundamental": True,
        "min_stars": 13,
        "indicators": {
            "primary":   [("EMA200_disc", 15.0)],
            "secondary": [],   # osciladores desactivados en crash
        },
        "min_signals": 1,
        "rationale": "Crash: solo estructural. Osciladores dan señal demasiado pronto."
    }
}

def calculate_confidence(indicators_triggered: list, regime: str, fundamental_ok: bool) -> float:
    """
    El confidence score no es un promedio simple.
    Cada indicador adicional que confirma suma menos que el anterior
    (rendimientos decrecientes de confirmación).
    
    Base: señal primaria = 50 puntos
    Primer confirmador: +25 puntos
    Segundo confirmador: +15 puntos
    Tercer confirmador: +10 puntos
    Fundamental ok: +20 puntos bonus
    Régimen SLOW_BEAR: +10 puntos bonus (ambiente favorable)
    
    Máximo teórico: 130 puntos -> normalizar a 100
    """
    if not indicators_triggered:
        return 0.0
        
    score = 0.0
    
    # primary
    score += 50
    
    # secondaries
    num_secondaries = len(indicators_triggered) - 1
    if num_secondaries >= 1:
        score += 25
    if num_secondaries >= 2:
        score += 15
    if num_secondaries >= 3:
        score += 10
        
    if fundamental_ok:
        score += 20
        
    if regime == "SLOW_BEAR":
        score += 10
        
    # Normalizar (max posible 50+25+15+10+20+10 = 130)
    normalized = (score / 130.0) * 100.0
    return min(100.0, normalized)

def evaluate_signal(indicator_values: dict, regime: str, fundamental_stars: int) -> dict:
    """
    Evalúa si un ticker tiene señal de compra según el régimen activo y sus valores técnicos actuales.
    `indicator_values` es un dict { "EMA200_disc": val, "BB_pctB": val, ... } con el valor del día.
    """
    if regime not in RULE_SETS:
        regime = "CALM"
        
    rules = RULE_SETS[regime]
    
    fundamental_ok = fundamental_stars >= rules["min_stars"]
    if rules["requires_fundamental"] and not fundamental_ok:
        return {
            "signal": False,
            "regime": regime,
            "rule_set": rules,
            "indicators_active": [],
            "indicators_triggered": [],
            "fundamental_ok": fundamental_ok,
            "confidence": 0.0,
            "rationale": f"Falla filtro fundamental ({fundamental_stars} < {rules['min_stars']})"
        }
        
    indicators_triggered = []
    
    # Check primary (solo asumimos 1 por ahora o requerimos que pase alguno)
    primary_passed = False
    for ind_name, th in rules["indicators"]["primary"]:
        val = indicator_values.get(ind_name)
        if val is None or pd.isna(val): continue
        # Determinar dirección
        direction = 'above' if ind_name == 'EMA200_disc' else 'below'
        if direction == 'above' and val >= th:
            primary_passed = True
            indicators_triggered.append(ind_name)
        elif direction == 'below' and val <= th:
            primary_passed = True
            indicators_triggered.append(ind_name)
            
    if not primary_passed:
        return {
            "signal": False,
            "regime": regime,
            "rule_set": rules,
            "indicators_active": [],
            "indicators_triggered": [],
            "fundamental_ok": fundamental_ok,
            "confidence": 0.0,
            "rationale": "No cumple señal primaria"
        }
        
    # Check secondaries
    for ind_name, th in rules["indicators"]["secondary"]:
        val = indicator_values.get(ind_name)
        if val is None or pd.isna(val): continue
        direction = 'above' if ind_name == 'EMA200_disc' else 'below'
        if direction == 'above' and val >= th:
            indicators_triggered.append(ind_name)
        elif direction == 'below' and val <= th:
            indicators_triggered.append(ind_name)
            
    num_signals = len(indicators_triggered)
    if num_signals < rules["min_signals"]:
        return {
            "signal": False,
            "regime": regime,
            "rule_set": rules,
            "indicators_active": rules["indicators"]["primary"] + rules["indicators"]["secondary"],
            "indicators_triggered": indicators_triggered,
            "fundamental_ok": fundamental_ok,
            "confidence": 0.0,
            "rationale": f"No alcanza señales mínimas ({num_signals} < {rules['min_signals']})"
        }
        
    confidence = calculate_confidence(indicators_triggered, regime, fundamental_ok)
    
    return {
        "signal": True,
        "regime": regime,
        "rule_set": rules,
        "indicators_active": rules["indicators"]["primary"] + rules["indicators"]["secondary"],
        "indicators_triggered": indicators_triggered,
        "fundamental_ok": fundamental_ok,
        "confidence": confidence,
        "rationale": rules["rationale"]
    }
