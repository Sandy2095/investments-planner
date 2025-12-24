from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import random
import os
from datetime import datetime

import csv

app = FastAPI(title="Investment Plan App")

templates = Jinja2Templates(directory="templates")

def load_instruments():
    instruments = []
    try:
        file_path = os.path.join("misc", "investment_declare.csv")
        if os.path.exists(file_path):
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean up keys/values if necessary
                    instruments.append(row)
    except Exception as e:
        print(f"Error loading instruments: {e}")
    return instruments

# Cache instruments
INSTRUMENTS_DATA = load_instruments()

def format_indian_number(value: float) -> str:
    try:
        value = int(float(value)) if value else 0
    except:
        return "0"
        
    s = str(value)
    if len(s) <= 3:
        return s
        
    last_three = s[-3:]
    remaining = s[:-3]
    
    formatted = ""
    while len(remaining) > 2:
        formatted = "," + remaining[-2:] + formatted
        remaining = remaining[:-2]
        
    formatted = remaining + formatted + "," + last_three
    return formatted

templates.env.filters["indian_format"] = format_indian_number

# --- Models ---
class InvestmentBreakdown(BaseModel):
    ppf: float = 0.0
    ssy: float = 0.0
    debt_rd: float = 0.0
    gold: float = 0.0
    emergency_fund: float = 0.0
    equity: float = 0.0
    risk_debt: float = 0.0

class Projection(BaseModel):
    target_year: int
    total_invested: float
    estimated_corpus: float
    years: int

class InvestmentPlan(BaseModel):
    total_investment_amount: float
    safe_amount: float
    risk_amount: float
    breakdown: InvestmentBreakdown
    projection: Projection
    notes: List[str]

# --- Helpers ---
def calculate_sip_fv(monthly_amount: float, annual_rate_percent: float, months: int) -> float:
    if monthly_amount <= 0 or months <= 0:
        return 0.0
    r = annual_rate_percent / 100 / 12
    # FV = P * ((1+r)^n - 1) * (1+r)/r
    fv = monthly_amount * (((1 + r) ** months - 1) * (1 + r)) / r
    return fv

def calculate_corpus_projection(breakdown: InvestmentBreakdown, target_year: int) -> Projection:
    current_year = datetime.now().year
    years_to_grow = target_year - current_year
    if years_to_grow < 1: years_to_grow = 1
    months = years_to_grow * 12
    
    # Assumptions
    # Equity: 12%
    # Risk Debt: 8%
    # Safe (SSY/PPF/RD): 7.5%
    # Gold: 8%
    # Emergency: 6% (Liquid)
    # SSY: 8.2% (Explicit)
    
    fv_equity = calculate_sip_fv(breakdown.equity, 12.0, months)
    fv_risk_debt = calculate_sip_fv(breakdown.risk_debt, 8.0, months)
    
    # Split Safe Bucket for accuracy if possible, but for now we aggregate mostly
    # SSY is usually higher rate (8.2%)
    fv_ssy = calculate_sip_fv(breakdown.ssy, 8.2, months)
    fv_ppf_rd = calculate_sip_fv(breakdown.ppf + breakdown.debt_rd, 7.1, months) # Averaging PPF/RD around 7.1
    
    fv_gold = calculate_sip_fv(breakdown.gold, 8.0, months)
    fv_emergency = calculate_sip_fv(breakdown.emergency_fund, 6.0, months)
    
    total_projected = fv_equity + fv_risk_debt + fv_ssy + fv_ppf_rd + fv_gold + fv_emergency
    
    # Total monthly investment
    total_monthly = (breakdown.ppf + breakdown.ssy + breakdown.debt_rd + 
                     breakdown.gold + breakdown.emergency_fund + 
                     breakdown.equity + breakdown.risk_debt)
                     
    return Projection(
        target_year=target_year,
        years=years_to_grow,
        total_invested=total_monthly * months,
        estimated_corpus=total_projected
    )

def get_random_quote() -> Dict[str, str]:
    try:
        file_path = os.path.join("misc", "quotes.txt")
        if not os.path.exists(file_path):
            return {"text": "Investment is a marathon, not a sprint.", "author": "AI Proverb"}
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Split by double newlines or based on logic
        raw_quotes = content.split('\n\n')
        parsed_quotes = []
        
        for q in raw_quotes:
            lines = q.strip().split('\n')
            if len(lines) >= 2:
                text = lines[0].strip()
                author = lines[1].strip()
                parsed_quotes.append({"text": text, "author": author})
                
        if parsed_quotes:
            return random.choice(parsed_quotes)
            
    except Exception as e:
        print(f"Error reading quotes: {e}")
        
    return {"text": "The best time to plant a tree was 20 years ago. The second best time is now.", "author": "Chinese Proverb"}

# --- Core Logic ---
def generate_auto_plan(
    salary: float, 
    age: int, 
    expenses: float, 
    emergency_fund: float, 
    high_risk_percent: float,
    girl_child_ages: List[int],
    target_year: int
) -> InvestmentPlan:
    # --- 1. Investment Capacity ---
    total_investment = salary - expenses
    if total_investment < 0: 
        total_investment = 0
    
    # --- 2. Bucket Split ---
    high_risk_p = high_risk_percent if high_risk_percent > 1 else high_risk_percent * 100
    risk_amount = total_investment * (high_risk_p / 100)
    safe_amount = total_investment * ((100 - high_risk_p) / 100)
    
    # --- 3. Emergency Fund ---
    emergency_target = expenses * 6
    emergency_shortfall = emergency_target - emergency_fund
    emergency_contribution = 0.0
    if emergency_shortfall > 0:
        # Invest 25% of safe bucket into emergency fund until full
        emergency_contribution = safe_amount * 0.25

    # --- 4. Safe Allocation Strategy ---
    # Net available for instruments after emergency contribution
    available_for_instruments = safe_amount - emergency_contribution
    if available_for_instruments < 0: available_for_instruments = 0

    # SSY Check
    eligible_ssy_count = sum(1 for a in girl_child_ages if a <= 10)
    ssy_amount = 0.0
    
    if eligible_ssy_count > 0:
        # Cap at 1.5L/yr per child (~12500/mo)
        total_ssy_demand = 12500 * eligible_ssy_count
        if available_for_instruments >= total_ssy_demand:
            ssy_amount = total_ssy_demand
        else:
            ssy_amount = available_for_instruments
            
    remaining_safe = available_for_instruments - ssy_amount
    
    # Standard Split for Remaining Safe: PPF (40%), Debt (35%), Gold (25%)
    ppf_amount = 0.0
    debt_rd_amount = 0.0
    gold_amount = 0.0
    
    if remaining_safe > 0:
        ppf_amount = remaining_safe * 0.40
        debt_rd_amount = remaining_safe * 0.35
        gold_amount = remaining_safe * 0.25
        
    # --- 5. Risk Allocation Strategy ---
    equity_amount = 0.0
    risk_debt_amount = 0.0
    
    if age < 35:
        # Aggressive: 100% Equity in Risk Bucket
        equity_amount = risk_amount
    else:
        # Moderate: 70% Equity, 30% Risk Debt
        equity_amount = risk_amount * 0.70
        risk_debt_amount = risk_amount * 0.30

    # Strategy Notes Logic
    notes = []
    
    # 1. Age-based Advice
    if age < 30:
        notes.append("Age Benefit: You are young! Consider increasing Risk Bucket allocation for higher long-term growth.")
    elif age > 50:
        notes.append("Age Caution: As you approach retirement, consider shifting more towards Safe Bucket to preserve capital.")
    else:
        notes.append("Balanced Approach: Maintain a healthy mix of growth and stability.")

    # 2. Allocation Breakdown
    notes.append("--- Allocation Summary ---")
    if total_investment > 0:
        notes.append(f"Safe Bucket: ₹{safe_amount:,.0f} ({safe_amount/total_investment*100:.1f}%)")
        if emergency_contribution > 0:
            notes.append(f"  - Emergency Fund: ₹{emergency_contribution:,.0f}")
        if ssy_amount > 0:
             notes.append(f"  - SSY (Priority): ₹{ssy_amount:,.0f}")
        notes.append(f"  - PPF/Debt/Gold: ₹{remaining_safe:,.0f}")
        
        notes.append(f"Risk Bucket: ₹{risk_amount:,.0f} ({risk_amount/total_investment*100:.1f}%)")
        if equity_amount > 0:
             notes.append(f"  - Equity/Mutual Funds: ₹{equity_amount:,.0f}")
    
    # 3. SSY Specific Note
    if eligible_ssy_count > 0:
        notes.append(f"SSY Strategy: Maximizing tax-free returns for {eligible_ssy_count} girl child(ren).")
    
    # 4. General Tip
    notes.append("Pro Tip: If your salary increases, try to invest 50% of the increment.")
    
    breakdown = InvestmentBreakdown(
        ppf=ppf_amount,
        ssy=ssy_amount,
        debt_rd=debt_rd_amount,
        gold=gold_amount,
        emergency_fund=emergency_contribution,
        equity=equity_amount,
        risk_debt=risk_debt_amount
    )
    
    projection = calculate_corpus_projection(breakdown, target_year)
    
    return InvestmentPlan(
        total_investment_amount=total_investment,
        safe_amount=safe_amount, # Derived
        risk_amount=risk_amount, # Derived
        breakdown=breakdown,
        projection=projection,
        notes=notes
    )

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    quote = get_random_quote()
    # Default values
    return templates.TemplateResponse("index.html", {
        "request": request,
        "plan": None,
        "quote": quote,
        "form_data": {
            "monthly_salary": 50000,
            "age": 22,
            "monthly_expenses": 20000,
            "current_emergency_fund": 0,
            "high_risk_percent": 60, # Default
            "target_year": 2040,
            "has_girl_child": False
        },
        "instruments": INSTRUMENTS_DATA,
        "min_year": datetime.now().year + 1
    })

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    # Main Inputs
    monthly_salary: float = Form(...),
    age: int = Form(...),
    monthly_expenses: float = Form(...),
    current_emergency_fund: float = Form(0),
    high_risk_percent: float = Form(50),
    target_year: int = Form(...),
    
    # Girl Child Inputs
    girl_child_count: int = Form(0),
    girl_child_age_1: Optional[int] = Form(None),
    girl_child_age_2: Optional[int] = Form(None),
    
    # SSY Calculator Inputs
    ssy_calc_amount: float = Form(0),
    ssy_girl_age: int = Form(0), 
    ssy_start_year: int = Form(0),
    
    # Action Type
    action: str = Form("auto"),
    
    # Manual Overrides (Optional)
    manual_ssy: float = Form(0),
    manual_ppf: float = Form(0),
    manual_debt: float = Form(0),
    manual_gold: float = Form(0),
    manual_emergency: float = Form(0),
    manual_equity: float = Form(0),
    manual_risk_debt: float = Form(0),
):
    if monthly_expenses > monthly_salary:
        quote = get_random_quote()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "plan": None,
            "quote": quote,
            "error_message": "Monthly Expenses cannot exceed Monthly Salary. Please adjust your inputs.",
            "form_data": {
                "monthly_salary": monthly_salary,
                "age": age,
                "monthly_expenses": monthly_expenses,
                "current_emergency_fund": current_emergency_fund,
                "high_risk_percent": high_risk_percent,
                "target_year": target_year,
                
                "girl_child_count": girl_child_count,
                "girl_child_age_1": girl_child_age_1,
                "girl_child_age_2": girl_child_age_2
            }
        })

    if action == "manual":
        # Create breakdown from inputs
        breakdown = InvestmentBreakdown(
            ssy=manual_ssy,
            ppf=manual_ppf,
            debt_rd=manual_debt,
            gold=manual_gold,
            emergency_fund=manual_emergency,
            equity=manual_equity,
            risk_debt=manual_risk_debt
        )
        
        # Recalculate Totals
        safe_sum = breakdown.ssy + breakdown.ppf + breakdown.debt_rd + breakdown.gold + breakdown.emergency_fund
        risk_sum = breakdown.equity + breakdown.risk_debt
        total_sum = safe_sum + risk_sum
        
        projection = calculate_corpus_projection(breakdown, target_year)
        
        plan = InvestmentPlan(
            total_investment_amount=total_sum,
            safe_amount=safe_sum,
            risk_amount=risk_sum,
            breakdown=breakdown,
            projection=projection,
            notes=["Manual Calculation - Rules may not apply."]
        )
        
    else: 
        # Auto Mode (Default)
        
        # Construct Girl Child Ages List
        girl_child_ages = []
        if girl_child_count >= 1 and girl_child_age_1 is not None:
             girl_child_ages.append(girl_child_age_1)
        if girl_child_count >= 2 and girl_child_age_2 is not None:
             girl_child_ages.append(girl_child_age_2)
        
        plan = generate_auto_plan(
            monthly_salary, age, monthly_expenses, current_emergency_fund, 
            high_risk_percent, girl_child_ages, target_year
        )
        
        # Recalculate projection with new breakdown
        plan.projection = calculate_corpus_projection(plan.breakdown, target_year)

    # Don't show quote on post back? Or show it but user has likely seen it.
    quote = get_random_quote()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "plan": plan,
        "quote": quote,
        "form_data": {
            "monthly_salary": monthly_salary,
            "age": age,
            "monthly_expenses": monthly_expenses,
            "current_emergency_fund": current_emergency_fund,
            "high_risk_percent": high_risk_percent,
            "target_year": target_year,
            
            "girl_child_count": girl_child_count,
            "girl_child_age_1": girl_child_age_1,
            "girl_child_age_2": girl_child_age_2
        },
        "instruments": INSTRUMENTS_DATA,
        "min_year": datetime.now().year + 1
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
