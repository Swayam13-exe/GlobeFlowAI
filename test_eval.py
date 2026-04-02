import sys
import os
sys.path.insert(0, os.getcwd())

from env.environment import WorkforceEnv
from env.models import Action

def test_easy_range_success():
    print("--- Test: Easy Range Success (In-Range) ---")
    env = WorkforceEnv()
    env.reset('easy')
    
    # Execute full correct sequence for Germany Easy
    actions = [
        ('request_document', 'passport'),
        ('verify_document', 'passport'),
        ('request_document', 'visa'),
        ('verify_document', 'visa'),
        ('request_document', 'employment_letter'),
        ('verify_document', 'employment_letter'),
        ('request_document', 'degree_certificate'),
        ('verify_document', 'degree_certificate'),
        ('approve_hr', 'HR'),
        ('approve_legal', 'Legal'),
        ('set_payroll', 'Germany'),
        ('set_tax_id', 'Germany'),
    ]
    
    for atype, target in actions:
        res = env.step(Action(action_type=atype, target=target))
        if res.info.get('result') != 'success':
            print(f"STEP FAILED: {atype}:{target} | Result: {res.info.get('result')} | Error: {res.info.get('error')}")
        else:
            print(f"STEP SUCCESS: {atype}:{target}")
            
    res = env.step(Action(action_type='finalize_case', target='all'))
    if res.info.get('status') != 'success':
        print(f"FINALIZE FAILED. Result: {res.info.get('result')} | Error: {res.info.get('error')} | Blockers: {res.info.get('blockers')}")
        print(f"Detail: {res.info.get('detail')}")
    
    assert res.info.get('status') == 'success'
    assert res.done == True
    print("OK\n")

def test_medium_overoptimization_fail():
    print("--- Test: Medium Over-optimization (Perfect Score -> Fail) ---")
    env = WorkforceEnv()
    env.reset('medium')
    
    # Medium expected range is [0.4, 0.8]. A perfect 1.0 should FAIL.
    # We'll clear all blockers correctly for Singapore.
    actions = [
        ('request_document', 'passport'),
        ('verify_document', 'passport'),
        ('request_document', 'visa'),
        ('verify_document', 'visa'),
        ('request_document', 'employment_letter'),
        ('verify_document', 'employment_letter'),
        ('approve_hr', 'HR'),
        ('approve_legal', 'Legal'),
        ('set_payroll', 'Singapore'),
        ('set_tax_id', 'Singapore'),
        ('set_shadow_payroll', 'Singapore'),
        ('set_pdpa', 'Singapore'),
        ('approve_finance', 'Finance'),
    ]
    
    for atype, target in actions:
        res = env.step(Action(action_type=atype, target=target))
        if res.info.get('result') != 'success':
            print(f"STEP FAILED: {atype}:{target} | Result: {res.info.get('result')} | Error: {res.info.get('error')}")
        else:
            print(f"STEP SUCCESS: {atype}:{target}")
        
    res = env.step(Action(action_type='finalize_case', target='all'))
    # A perfect score for Singapore Medium will be 1.10 (with bonus), 
    # which is > 0.80.
    print(f"FINALIZE Result: {res.info.get('result')} | Status: {res.info.get('status')} | Score: {res.info.get('final_score')}")
    assert res.info.get('status') == 'failed'
    assert res.done == True
    print("OK\n")

def test_hard_uae_fail():
    print("--- Test: Hard (UAE No-Tax Trap) ---")
    env = WorkforceEnv()
    env.reset('hard')
    
    # Critical penalty: UAE set_tax_id
    env.step(Action(action_type='set_tax_id', target='UAE'))
    
    # Attempt to finalize
    res = env.step(Action(action_type='finalize_case', target='all'))
    print(f"FINALIZE Result: {res.info.get('result')} | Status: {res.info.get('status')} | Score: {res.info.get('final_score')}")
    # Hard range is [0.2, 0.6]. A score with a -0.3 penalty will be low.
    # Also blockers will remain since we didn't clear them.
    assert res.info.get('status') == 'failed' or res.info.get('result') == 'rule_violation'
    print("OK\n")

if __name__ == "__main__":
    try:
        test_easy_range_success()
        test_medium_overoptimization_fail()
        print("=== ALL EVALUATION TESTS PASSED ===")
    except Exception as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
