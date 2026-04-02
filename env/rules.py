RULES = {
    "Germany": {
        "requires_visa": True,
        "requires_tax": True,
        "requires_payroll": False,
        "requires_pdpa": False,
        "requires_shadow_payroll": False
    },
    "Singapore": {
        "requires_visa": True,
        "requires_tax": False,
        "requires_payroll": True,
        "requires_pdpa": True,
        "requires_shadow_payroll": True
    },
    "UAE": {
        "requires_visa": True,
        "requires_tax": False,
        "requires_payroll": True,
        "requires_pdpa": False,
        "requires_shadow_payroll": False
    }
}

def get_country_rules(countries: list[str]) -> dict:
    combined_rules = {
        "requires_visa": False,
        "requires_tax": False,
        "requires_payroll": False,
        "requires_pdpa": False,
        "requires_shadow_payroll": False,
        "tax_forbidden": False,
    }
    
    for country in countries:
        if country not in RULES:
            continue
            
        rules = RULES[country]
        if rules["requires_visa"]:
            combined_rules["requires_visa"] = True
        if rules["requires_tax"]:
            combined_rules["requires_tax"] = True
        if rules["requires_payroll"]:
            combined_rules["requires_payroll"] = True
        if rules["requires_pdpa"]:
            combined_rules["requires_pdpa"] = True
        if rules["requires_shadow_payroll"]:
            combined_rules["requires_shadow_payroll"] = True
            
        if country == "UAE":
            combined_rules["tax_forbidden"] = True
            
    return combined_rules
