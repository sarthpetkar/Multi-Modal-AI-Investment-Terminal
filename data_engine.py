import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def categorize_market_cap(market_cap_cr):
    """Classifies company size based on standard Indian market definitions."""
    if market_cap_cr >= 20000:
        return "Large Cap (Blue Chip, Stable)"
    elif 5000 <= market_cap_cr < 20000:
        return "Mid Cap (Growth Phase)"
    elif 500 <= market_cap_cr < 5000:
        return "Small Cap (High Growth / High Risk)"
    else:
        return "Micro Cap (Extremely Volatile)"

def get_financial_metrics(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        # --- 1. BASE METRICS EXTRACTION ---
        market_cap_cr = info.get('marketCap', 0) / 10000000
        company_type = categorize_market_cap(market_cap_cr)
        
        # Existing metrics
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        opm = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
        debt_to_equity = info.get('debtToEquity', 100) / 100 if info.get('debtToEquity') else 100
        current_ratio = info.get('currentRatio', 0)
        pe_ratio = info.get('trailingPE', 100)
        peg_ratio = info.get('pegRatio', 100)
        fcf_cr = info.get('freeCashflow', 0) / 10000000 if info.get('freeCashflow') else 0
        # The yfinance Dividend Glitch Fix
        raw_div = info.get('dividendYield', 0)
        div_yield = raw_div * 100 if raw_div else 0
        if div_yield > 100: 
            div_yield = raw_div
        promoter_holding = info.get('heldPercentInsiders', 0) * 100 if info.get('heldPercentInsiders') else 0
        revenue = info.get('totalRevenue', 0)
        book_value = info.get('bookValue', 0)
        eps = info.get('trailingEps', 0)
        roa = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        low_52 = info.get('fiftyTwoWeekLow', 0)
        current_price = info.get('currentPrice', 0)
        
        # --- 2. COMPARATIVE EVALUATION MATRIX ---
        score = 0
        max_criteria = 12
        evaluation_list = []
        
        # ... (keep the helper function and Cap-specific rules the same) ...

        
        
        def evaluate(metric, actual, target_val, operator_str, format_type="num"):
            nonlocal score
            
            # Determine Pass/Fail based on operator
            if operator_str == ">":
                passed = actual > target_val
                variance = ((actual - target_val) / abs(target_val)) * 100 if target_val != 0 else 0
                target_display = f"> {target_val}"
            elif operator_str == "<":
                passed = actual < target_val
                # For "less than" targets, being lower is a positive variance
                variance = ((target_val - actual) / abs(target_val)) * 100 if target_val != 0 else 0
                target_display = f"< {target_val}"
                
            if passed: score += 1
            
            # Format the Actual Value
            if format_type == "pct": 
                actual_str = f"{actual:.2f}%"
                target_display += "%"
            elif format_type == "ratio": 
                actual_str = f"{actual:.2f}"
            else: 
                actual_str = f"₹ {actual:,.0f}"

            # THE UPGRADE: Smart Variance Formatting
            # Suppress variance for zero-targets (FCF/EPS) and size filters (Market Cap)
            if target_val == 0 or "Market Cap" in metric:
                var_str = "-"
            else:
                if variance > 0:
                    var_str = f"+{variance:,.1f}%"
                else:
                    var_str = f"{variance:,.1f}%"
                
            evaluation_list.append({
                "Fundamental Metric": metric,
                "Actual Value": actual_str,
                "Institutional Target": target_display,
                "Variance": var_str,
                "Status": "✅ Pass" if passed else "❌ Fail"
            })

        # --- 4-TIER DYNAMIC CAP LOGIC ---
        if "Large Cap" in company_type:
            evaluate("Return on Equity (ROE)", roe, 15.0, ">", "pct")
            evaluate("Operating Margin (OPM)", opm, 12.0, ">", "pct")
            evaluate("Debt to Equity", debt_to_equity, 1.0, "<", "ratio")
            evaluate("P/E Ratio", pe_ratio, 30.0, "<", "ratio")
            
        elif "Mid Cap" in company_type:
            evaluate("Return on Equity (ROE)", roe, 18.0, ">", "pct")
            evaluate("Operating Margin (OPM)", opm, 14.0, ">", "pct")
            evaluate("Debt to Equity", debt_to_equity, 0.6, "<", "ratio")
            evaluate("P/E Ratio", pe_ratio, 25.0, "<", "ratio")
            
        elif "Small Cap" in company_type:
            evaluate("Return on Equity (ROE)", roe, 20.0, ">", "pct")
            evaluate("Operating Margin (OPM)", opm, 15.0, ">", "pct")
            evaluate("Debt to Equity", debt_to_equity, 0.4, "<", "ratio")
            evaluate("P/E Ratio", pe_ratio, 20.0, "<", "ratio")
            
        else: # Micro Cap (Extremely Strict)
            evaluate("Return on Equity (ROE)", roe, 25.0, ">", "pct")
            evaluate("Operating Margin (OPM)", opm, 18.0, ">", "pct")
            evaluate("Debt to Equity", debt_to_equity, 0.1, "<", "ratio") # Almost zero debt allowed
            evaluate("P/E Ratio", pe_ratio, 15.0, "<", "ratio") # Must be exceptionally cheap

        # Updated Universal Rules
        evaluate("Current Ratio", current_ratio, 1.2, ">", "ratio")
        evaluate("PEG Ratio", peg_ratio, 1.0, "<", "ratio")
        # Update these two lines in your Universal Rules section
        evaluate("Free Cash Flow (Cr)", fcf_cr, 0, ">", "num")
        evaluate("Market Cap Stability (Cr)", market_cap_cr, 500.0, ">", "num")
        evaluate("Promoter Holding", promoter_holding, 50.0, ">", "pct")
        evaluate("Return on Assets (ROA)", roa, 10.0, ">", "pct") # >10% is a standard baseline
        evaluate("Dividend Yield", div_yield, 1.0, ">", "pct") # Proves cash return to shareholders
        evaluate("Earnings Per Share (EPS)", eps, 0, ">", "num") # Must be profitable

        df_metrics = pd.DataFrame(evaluation_list)
        moat_pct = round((score / max_criteria) * 100, 1)

        # --- 3. FORENSIC BALANCE SHEET AUDIT ---
        bs = stock.balance_sheet
        cf = stock.cashflow
        inc = stock.financials
        
        recent_bs = bs.iloc[:, 0] if not bs.empty else pd.Series()
        recent_cf = cf.iloc[:, 0] if not cf.empty else pd.Series()
        recent_inc = inc.iloc[:, 0] if not inc.empty else pd.Series()

        working_capital = recent_bs.get("Total Current Assets", 0) - recent_bs.get("Total Current Liabilities", 0)
        retained_earnings = recent_bs.get("Retained Earnings", 0)
        ebit = recent_inc.get("EBIT", 0)
        total_assets = recent_bs.get("Total Assets", 1) 
        total_liabilities = recent_bs.get("Total Liabilities Net Minority Interest", 1)
        
        z_score = 0
        if total_assets > 1:
            t1 = working_capital / total_assets
            t2 = retained_earnings / total_assets
            t3 = ebit / total_assets
            t4 = (market_cap_cr * 10000000) / total_liabilities 
            z_score = (1.2 * t1) + (1.4 * t2) + (3.3 * t3) + (0.6 * t4)

        net_income = recent_inc.get("Net Income", 1)
        operating_cf = recent_cf.get("Operating Cash Flow", 0)
        earnings_quality = operating_cf / net_income if net_income > 0 else 0

        forensic_list = [
            {
                "Forensic Check": "Altman Z-Score (Bankruptcy Risk)",
                "Value": f"{z_score:.2f}",
                "Institutional Standard": "> 2.99 (Safe)",
                "Flag": "🟢 Safe Zone" if z_score > 2.99 else "🔴 Distress Zone"
            },
            {
                "Forensic Check": "Earnings Quality (OCF / Net Income)",
                "Value": f"{earnings_quality:.2f}x",
                "Institutional Standard": "> 1.0x (Cash-backed)",
                "Flag": "🟢 Verified" if earnings_quality > 1.0 else "🔴 Red Flag (Paper Profits)"
            }
        ]
        df_forensics = pd.DataFrame(forensic_list)


        # --- NEW: ADVANCED BLENDED VALUATION ---
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # 1. Graham Number (Asset-Heavy Defensive Value)
        graham_value = (22.5 * eps * bvps) ** 0.5 if eps > 0 and bvps > 0 else 0
        
        # 2. Earnings Power Value (Using Forward P/E or default 15)
        forward_pe = info.get('forwardPE', 15) # Defaults to 15 if missing
        earnings_value = eps * forward_pe if eps > 0 else 0
        
        # 3. Wall Street Consensus (Analyst Mean Target)
        target_mean = info.get('targetMeanPrice', 0)

        # Calculate Blended Fair Value (Average of all valid models)
        valid_models = [v for v in [graham_value, earnings_value, target_mean] if v > 0]
        blended_fair_value = sum(valid_models) / len(valid_models) if valid_models else 0
            
        margin_of_safety = ((blended_fair_value - current_price) / blended_fair_value) * 100 if blended_fair_value > 0 else 0

        valuation_dict = {
            "Current Price": current_price,
            "Graham Value": graham_value,
            "Earnings Value": earnings_value,
            "Target Price": target_mean,
            "Blended Fair Value": blended_fair_value,
            "Margin of Safety (%)": margin_of_safety,
            "52W High": info.get('fiftyTwoWeekHigh', 0),
            "52W Low": info.get('fiftyTwoWeekLow', 0),
            "P/E": info.get('trailingPE', 0),
            "P/B": info.get('priceToBook', 0),
            "Div Yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }

        # --- NEW: RAW FINANCIAL STATEMENTS & SHAREHOLDINGS ---
        financial_statements = {
            "Profit & Loss": stock.financials,
            "Balance Sheet": bs, # Re-using the 'bs' variable we pulled for forensics
            "Cash Flow": cf,     # Re-using the 'cf' variable
            "Major Holders": stock.major_holders,
            "Institutional Holders": stock.institutional_holders
        }

        # --- 4. PACKAGE FOR ML ENGINE & UI ---
        raw_dict = {
            "Company Type": company_type,
            "Moat Score (%)": moat_pct,
            "Criteria Passed": f"{score}/{max_criteria}"
        }

        # --- NEW: DUPONT ANALYSIS (ROE DECONSTRUCTION) ---
        net_income = info.get('netIncomeToCommon', 0)
        total_rev = info.get('totalRevenue', 0)
        total_assets = info.get('totalAssets', 0)
        total_equity = info.get('totalStockholderEquity', 0)
        
        # 1. Net Profit Margin (Profitability)
        npm = (net_income / total_rev) * 100 if total_rev > 0 else 0
        # 2. Asset Turnover (Operational Efficiency)
        asset_turnover = (total_rev / total_assets) if total_assets > 0 else 0
        # 3. Equity Multiplier (Financial Leverage / Risk)
        equity_multiplier = (total_assets / total_equity) if total_equity > 0 else 0
        
        dupont_dict = {
            "Net Profit Margin": npm,
            "Asset Turnover": asset_turnover,
            "Equity Multiplier": equity_multiplier,
            "Calculated ROE": (npm / 100) * asset_turnover * equity_multiplier * 100
        }

        # --- NEW: NLP NEWS SENTIMENT ANALYSIS ---
        try:
            news_data = stock.news
            analyzer = SentimentIntensityAnalyzer()
            news_list = []
            total_compound = 0
            
            if news_data:
                for article in news_data[:10]: 
                    title = article.get('title', '')
                    
                    # THE FIX: If there is no title, skip this completely!
                    if not title or title.isspace():
                        continue
                        
                    publisher = article.get('publisher', 'Unknown')
                    link = article.get('link', '#')
                    
                    sentiment = analyzer.polarity_scores(title)
                    compound = sentiment['compound']
                    total_compound += compound
                    
                    if compound >= 0.05: status = "🟢 Bullish"
                    elif compound <= -0.05: status = "🔴 Bearish"
                    else: status = "⚪ Neutral"
                    
                    news_list.append({
                        "Title": title, "Publisher": publisher, 
                        "Sentiment": status, "Score": compound, "Link": link
                    })
            
            # Check if we actually found valid news
            if news_list:
                avg_score = total_compound / len(news_list)
                if avg_score >= 0.05: overall_sentiment = "Bullish"
                elif avg_score <= -0.05: overall_sentiment = "Bearish"
                else: overall_sentiment = "Neutral"
            else:
                # If no valid news was found, trigger the "No Data" state
                overall_sentiment = "No Data"
                avg_score = 0
            
            sentiment_dict = {
                "Overall Sentiment": overall_sentiment,
                "Average Score": avg_score,
                "News Articles": news_list
            }
        except Exception as e:
            sentiment_dict = {"Overall Sentiment": "Unavailable", "Average Score": 0, "News Articles": []}

        # --- 4. PACKAGE FOR ML ENGINE & UI ---
        raw_dict = {
            "Company Type": company_type,
            "Moat Score (%)": moat_pct,
            "Criteria Passed": f"{score}/{max_criteria}"
        }

        # Downloads 5 years of daily open/high/low/close/volume data
        hist_data = stock.history(period="5y")
        
        # CRITICAL UPDATE: We now return 9 items (Added sentiment_dict)
        return df_metrics, df_forensics, info, raw_dict, valuation_dict, financial_statements, dupont_dict, hist_data, sentiment_dict

    except Exception as e:
        return None, None, {"error": str(e)}, {}, {}, {}, {}, None, {} 

   