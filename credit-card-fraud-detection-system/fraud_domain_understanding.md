# The Real Story Behind the Data
## IEEE-CIS Fraud Detection — Deep Domain Understanding
**"You cannot fight what you don't understand."**

---

## PART 1 — The Real-World Context: What Is Actually Happening?

### Who Made This Data?

**Vesta Corporation** is a payment service company. They sit between:

```
[Customer] → [Merchant Website] → [Vesta] → [Card Network: Visa/MC] → [Issuing Bank]
```

When you buy something online, Vesta is the company processing and **guaranteeing** that transaction. They absorb the fraud loss themselves if they approve a fraudulent transaction. So detecting fraud is not academic for them — it is their **survival**. Every row in this dataset is a real moment where Vesta had to decide in milliseconds: *approve or decline*.

### What Does "Fraud" Actually Mean Here?

This is specifically **Card-Not-Present (CNP) fraud** — the kind that happens in online transactions where the physical card is never swiped. This is the most common and fastest-growing type of fraud globally.

**The 5 main ways this fraud happens in the real world:**

| Fraud Type | How It Works | What It Looks Like in Data |
|---|---|---|
| **Stolen Card Data** | Fraudster buys card numbers from dark web data breaches | New device, new location, name mismatch, first-time email |
| **Card Testing** | Fraudster runs \$1–\$5 test transactions to check if a stolen card is still active | Multiple small transactions, short time between them, same card different emails |
| **Account Takeover** | Fraudster logs into victim's shopping account and changes shipping address | Cookie "Found" (returning account) but new device, address change, large amount |
| **Synthetic Identity** | Fraudster creates a fake identity combining real + fake info | Mix of matching and non-matching M-flags, anonymous email, no prior history |
| **Reshipping Scam** | Fraudster ships goods to a "mule" address far from billing address | Large dist1/dist2, P and R email domains differ, high-value product |

---

## PART 2 — The Dataset Architecture

The dataset is split into two joined tables per set:

```
train_transaction  (590,540 rows × 394 cols)  ← the "what happened"
        │
        │  JOIN on TransactionID
        │
train_identity     (144,233 rows × 41 cols)   ← the "who and how" (device/browser/network)
```

**Only 24% of transactions have an identity record.** Why? Because Vesta only collects identity/device signals when the merchant's website supports their JavaScript SDK. If the merchant hasn't integrated it, there's no device fingerprint data — but the transaction still gets logged.

**This is itself a signal.** Transactions without identity records have a different fraud profile than those with records. Never throw away the "identity record present" flag.

---

## PART 3 — Transaction Table: Column-by-Column Truth

---

### `TransactionID`
**What it is:** A unique surrogate key for every transaction. Purely an identifier.  
**Fraud relevance:** On its own, nothing. But when you group by `TransactionID` to link with identity, you can see which transactions have device fingerprints and which don't.

---

### `isFraud`
**What it is:** The ground truth label. `1` = confirmed fraudulent, `0` = legitimate.  
**How Vesta knows:** After the transaction is approved, Vesta waits. If the cardholder later files a **chargeback** (disputes the charge with their bank), that transaction is retrospectively flagged as fraud. This means the labels are **delayed** — Vesta didn't know at the time; they found out later. This is why fraud detection is hard: you must predict the future from the past.

> **Key Insight:** The 3.5% fraud rate is not the "true" fraud rate. Some fraud is never reported (cardholder doesn't notice small charges). Real fraud rate is likely higher — but only confirmed chargebacks are labeled.

---

### `TransactionDT`
**What it is:** Time elapsed in **seconds** from a fixed reference point (not a real date — Vesta anonymized it for privacy). It's a relative timestamp.

**Why it matters — the fraud clock:**
- A fraudster who just bought stolen card data will use it **immediately** before the bank catches on
- Legitimate users transact at human hours; fraud bots transact at 3 AM
- `D1` (days since last transaction) derives from this — very short D1 means rapid repeated use

**What to engineer:** Convert to hour-of-day (0–23), day-of-week (Mon–Sun), and "days since training start." Fraud spikes in certain hours and weekends.

---

### `TransactionAmt`
**What it is:** The dollar amount of the transaction (in USD).

**The fraud pattern:**
- **Card testing** = very small amounts (\$0.25–\$5). A fraudster runs a micro-transaction first to verify the card works.
- **After testing** = suddenly a large amount (\$500–\$2000) on the same card minutes later.
- **Legit customers** have a history of normal spending on their card; a \$2000 transaction from a card that always spends \$30 is an anomaly.

**What to engineer:**
- `log1p(TransactionAmt)` — de-skew it
- Ratio of this transaction to the card's historical average amount
- Flag if amount > 3× the card's median

---

### `ProductCD` — Product Category Code
**What it is:** The category of what was purchased. There are 5 values:

| Code | Likely Meaning | Fraud Risk |
|---|---|---|
| **W** | Physical goods / "Walmart-style" retail (74.5% of data) | Medium — goods can be reshipped |
| **C** | Could be "Cash-equivalent" / gift cards | **High** — gift cards are untraceable and instantly redeemable |
| **H** | Hotel / travel booking or "Hardware" | Medium |
| **R** | Retail or "refund-related" | Medium |
| **S** | Services / subscriptions | Lower — harder to monetize stolen services |

> **Key Insight:** Vesta anonymized the exact meanings. But the fraud rate **differs significantly** across product codes. A model should treat `ProductCD` as a strong categorical feature, not just impute it as a number.

---

### `card1` through `card6` — The Card Identity
This is the most important feature group. It tells you **what card was used**.

| Column | What It Actually Is | Why It Matters |
|---|---|---|
| **card1** | Masked/hashed portion of the card number. Think of it as a card identifier | Lets you group all transactions on the same card. If card1 has 10 transactions in 1 hour, that's card-testing fraud |
| **card2** | Additional card attribute — likely related to the card verification number region or bank product code | Helps disambiguate cards with same card1 |
| **card3** | Numeric — likely the **billing zip code** area code or issuer bank routing code | Geographic grounding of the card |
| **card4** | Card network: Visa, Mastercard, Amex, Discover | Amex has stronger fraud controls; Discover is rare. Visa/MC are primary targets |
| **card5** | Numeric — likely the **card product type code** from the issuer (e.g., student card, platinum, business) | Business cards behave differently from personal cards |
| **card6** | Card type: **debit** or **credit** | Critical: debit cards draw from a real bank account — fraud is immediately painful. Credit cards have 30-day grace periods — fraud goes unnoticed longer. Fraudsters prefer credit cards because there's more time before the victim notices |

**The card fingerprint:** Together, `card1 + card2 + card3 + card4 + card5 + card6` gives you a near-unique card fingerprint without exposing actual card numbers. You should engineer features like: *how many distinct emails has this card1 used? How many distinct addresses? Over what time period?*

---

### `addr1` and `addr2` — Billing Address
| Column | What It Is | Fraud Signal |
|---|---|---|
| **addr1** | Billing zip/postal code (encoded numerically) | If the same zip code is used by thousands of different cards rapidly, it could be a mule address |
| **addr2** | Billing country code | If the card is issued in the US (based on card3) but addr2 says a foreign country, that's suspicious |

**The mismatch problem:** A stolen card will have the real cardholder's billing address. But a fraudster might use a different shipping address. `addr1` is the billing side — compare it against `dist1` to see how far the goods are being shipped.

---

### `dist1` and `dist2` — Distance Features
| Column | What It Is | Fraud Signal |
|---|---|---|
| **dist1** | Distance (in miles) between billing address and the shipping/delivery address | **High dist1 = reshipping scam.** A fraudster in Nigeria with a stolen US card ships goods to a US reshipping mule, then internationally. dist1 = 1500 miles is a red flag |
| **dist2** | Distance between billing address and the identity/home address on file | Also a mismatch signal; 93.6% missing because this requires identity verification data |

---

### `P_emaildomain` and `R_emaildomain` — Email Domains
| Column | What It Is | Fraud Signal |
|---|---|---|
| **P_emaildomain** | Email domain of the **Purchaser** (the person paying) | `anonymous.com` is huge red flag — deliberately hiding identity |
| **R_emaildomain** | Email domain of the **Recipient** (who receives the order confirmation) | If P and R email domains are completely different services, someone is redirecting the confirmation |

**The fraud story in emails:**
- Legitimate customer: P_email = `gmail.com`, R_email = `gmail.com` (same person)
- Fraudster: P_email = `anonymous.com`, R_email = `yahoo.com` (different identity, hiding)
- Account takeover: P_email = `gmail.com` (victim's), R_email = `protonmail.com` (fraudster's burner)

**Engineer:** Flag if P_emaildomain == R_emaildomain. Flag known disposable/anonymous email providers. Extract TLD (`.com`, `.ru`, `.cn`).

---

### `C1` through `C14` — Count Features (THE VELOCITY DETECTORS)
These are the **most important behavioral features** in the dataset. They count how many unique things are **associated with this payment card** up to the time of this transaction. They are Vesta's own aggregation of historical behavior.

| Column | What It Counts | Fraud Interpretation |
|---|---|---|
| **C1** | Number of distinct billing addresses ever used with this card | A legitimate person has 1–2 addresses (home, work). A fraudster cycling through cards might have 10+ different addresses |
| **C2** | Number of distinct email addresses used with this card | One card, one email = normal. One card, 50 emails = card being shared/used fraudulently |
| **C3** | Number of different cards used at the same billing address | Multiple cards at one address = possible mule address or a family — but 20+ cards = almost certainly a fraud hub |
| **C4** | Count of declined/failed transactions | High C4 = card-testing in progress right now |
| **C5** | Count of transactions where the email domain was "unfamiliar" to the system | New/anonymous email domains = identity hiding |
| **C6** | Count of email addresses on Vesta's "on-file" (trusted) list | Low C6 + high C2 = using many untrusted emails — suspicious |
| **C7** | Count of cards used in a recent time window | High count = card-testing machine — a fraudster bot rapidly testing many cards |
| **C8** | Count of orders in a recent time window | Multiple rapid orders = bot behavior or card-testing |
| **C9** | Count of distinct emails in a time window | Velocity of new email creation = fraud setup |
| **C10** | Count of billing addresses in a time window | Rapid address changes = fraud setup |
| **C11** | Lifetime count of cards associated with this account | 50 lifetime cards on one account = definitely fraud history |
| **C12** | Lifetime count of emails associated | Same logic as C2 but longer time horizon |
| **C13** | Lifetime count of addresses associated | Same logic as C1 but longer time horizon |
| **C14** | Count of something else (exact meaning undisclosed by Vesta) | Still a useful velocity/count signal |

> **The velocity story:** A real customer might have C1=1, C2=1, C7=1. A fraudster running 500 stolen cards per day will have C7=500, C8=500, C4=300 (300 declines, 200 approvals). These count features are Vesta watching the fraudster's behavior over time and recording it as a number.

---

### `D1` through `D15` — Delta/Days Features (THE TIME DETECTORS)
These measure **time gaps** between the current transaction and various historical events for this card/user.

| Column | What It Measures | Fraud Interpretation |
|---|---|---|
| **D1** | Days since the **last transaction** on this card | D1 = 0 or 0.001 → this card was just used minutes ago. Multiple transactions in rapid succession = card-testing. D1 = 500 → card dormant for 500 days, then suddenly a big transaction = stolen dormant card |
| **D2** | Days since the last transaction with the **same email** | If email was never used before (D2 = NaN or very large), it's a new/burner email |
| **D3** | Days since this **shipping address** was last used | New shipping address = possible reshipping destination |
| **D4** | Days since this **email address** was last used on Vesta's system | Brand-new email to Vesta's system = suspicious |
| **D5** | Days since this **recipient email** was last used | |
| **D6–D9** | Various other time deltas (93%+ missing — likely only captured in specific merchant integrations) | Less reliable but still useful when present |
| **D10** | Days since the last transaction with this **card on this device** | New device + old card = possible account takeover |
| **D11** | Days since this card was **first seen** by Vesta | A brand-new card that immediately makes large transactions = high risk |
| **D12–D15** | Other temporal deltas (mostly missing) | Use as binary "was this information available" features |

> **The dormant card attack:** A fraudster buys a batch of 5-year-old stolen card data. D11 = 1800 days (card first seen 5 years ago), D1 = 1800 days (last used 5 years ago). Suddenly it's being used today for \$800. That combination — very old card, very long gap, large amount — is a classic pattern.

---

### `M1` through `M9` — Match Flags (THE IDENTITY CHECKERS)
These are **binary match results** — Vesta comparing information provided in this transaction against information on file. Think of it as: "does what you're claiming match what we know about you?"

Values: `T` (True/Match), `F` (False/No-Match), or missing.

| Column | What Is Being Matched | Fraud Interpretation |
|---|---|---|
| **M1** | Name on card vs. billing name entered at checkout | M1 = F → the name entered doesn't match the card. Stolen card used by someone who doesn't know the real name |
| **M2** | Card verification / CVV2 area match | M2 = F → failed CVV verification — almost always attempted fraud |
| **M3** | Billing address match against card's registered address | M3 = F → address entered doesn't match what the issuing bank has on file |
| **M4** | Some categorical match (encoded as M0, M1, M2 values — a multi-class flag) | Different tiers of identity verification result |
| **M5** | Email address match vs. on-file email | M5 = F → unfamiliar email address for this card |
| **M6** | Phone or account flag | |
| **M7** | Additional match flag | When M7–M9 are all F on the same transaction, that's a pattern of systemic mismatch = identity fraud |
| **M8** | Additional match flag | |
| **M9** | Additional match flag | |

> **The mismatch cascade:** A real cardholder will have M1=T, M2=T, M3=T (everything matches because they are who they say they are). A fraudster with stolen card data might get M1=F (wrong name), M3=F (wrong address — they only have the card number not the billing address), M5=F (different email). Three or more F flags = near-certain fraud attempt.

---

### `V1` through `V339` — Vesta Proprietary Features (THE SECRET WEAPONS)

These 339 features are **engineered by Vesta's own data science team**. They are not raw fields — they are derived statistics, ratios, and signals computed from Vesta's internal transaction history and risk models. Vesta deliberately anonymized them so competitors can't replicate their fraud detection logic.

**What we know from research and the data structure:**

The V-features are organized in **correlated subgroups**, each linked to a specific C-feature:

| Subgroup | Linked to | What They Likely Represent |
|---|---|---|
| **V1–V11** | C1 (address count) | Derived statistics about address usage: how many transactions at this address, fraud rate at this zip, normalized counts |
| **V12–V34** | C2 (email count) | Email-based risk signals: age of email domain, normalized email velocity, email risk score |
| **V35–V52** | C4 (failure count) | Failure rate signals: ratio of declines to approvals, consecutive failure streak |
| **V53–V74** | C5 (unfamiliar emails) | Risk signals for unfamiliar email behavior |
| **V75–V94** | C6 (on-file emails) | Trust signals for known email associations |
| **V95–V137** | C9/C10 (velocity) | Velocity-normalized risk scores: this card vs. average card at this time of day |
| **V138–V166** | C11 (lifetime cards) | Long-term card network signals |
| **V167–V216** | C12 (lifetime emails) | Long-term email network signals |
| **V217–V278** | C13 (lifetime addresses) | Long-term address network signals |
| **V279–V321** | C14 | Additional network signals |
| **V322–V339** | Other | Miscellaneous engineered signals |

**How to treat V-features:**
- Within each subgroup, they are highly correlated — pick the best or use PCA within the group
- Many will be zero or NaN for new customers (no history = no computed signal)
- NaN in V-features is NOT random — it means "this subgroup of signals wasn't computable for this transaction" which is itself a signal
- Do NOT impute V-features with global mean — the subgroup structure means you'd mix signals from different semantic categories

---

## PART 4 — Identity Table: Column-by-Column Truth

The identity table answers the question: **"What device and network was used to make this transaction?"**

---

### `id_01` and `id_02` — Score Features
| Column | What It Is | Fraud Signal |
|---|---|---|
| **id_01** | A score — likely a **proxy for how "suspicious" the current session looks** to Vesta's real-time risk engine. Negative values are common (see sample: -45, 0, -5) | More negative = riskier session. A real customer browsing normally gets 0. A bot gets a large negative score |
| **id_02** | A second score — likely the **historical transaction count or account score** for this device/browser fingerprint. Very large values (70,000+) indicate high-activity accounts | Low id_02 on a device that claims to be a returning customer = fingerprint spoofing |

---

### `id_03` and `id_04` — Additional Scores
Numeric scores, 54% missing. When present, they capture additional session-level risk signals. The high missingness means only certain merchant integrations provide these.

---

### `id_05` and `id_06` — Behavioral Counts
Numeric. These count behavioral events in the current session (e.g., how many page clicks, how many form resubmissions). A bot will have unnatural patterns here — too fast, too regular.

---

### `id_07` and `id_08` — Screen/Browser Counts
Very sparse (96.4% missing). When present, these may capture screen resolution change events or browser capability probes — tests used to detect virtual machines or headless browsers (tools fraudsters use to automate attacks).

---

### `id_09` and `id_10` — Time/Event Counters
~48% missing. These may count events over a time window for this device. A bot making 1000 requests per hour will have very different id_09/id_10 values than a human who takes 3 minutes to fill out a form.

---

### `id_11` — Account/Network Score
Almost always `100.0`. This appears to be a network-level health score. When it deviates from 100, pay close attention.

---

### `id_12` — Billing Address Verification Result
Values: `Found` / `NotFound`  
**What it means:** When Vesta checks the billing address against their database — was it **ever seen before**?
- `Found` (14.7%) = this billing address is known to Vesta from previous transactions
- `NotFound` (85.3%) = this billing address has **never been seen** before on Vesta's network

> **Why NotFound dominates:** Most transactions are from customers new to this particular merchant. But for fraud detection, `NotFound` + other red flags = much higher risk. A fraudster using a random address will always be `NotFound`.

---

### `id_13` — Count Feature
Numeric. Likely the number of transactions from the same billing address on Vesta's network. Low value = new address. High value = well-established address (lower risk).

---

### `id_14` — Timezone Offset
**This is one of the most powerful hidden features.**  
Values: numeric minutes (e.g., -480, -300, -240)

This is the **UTC offset in minutes** of the user's browser:
- `-300` = UTC-5 = US Eastern Time
- `-480` = UTC-8 = US Pacific Time  
- `+330` = UTC+5:30 = India

**The fraud signal:** If a card's billing address is in New York (Eastern Time) but id_14 shows the browser is in UTC+8 (China/Southeast Asia), the person using the card is physically in a different country from where the card is registered. This is the digital equivalent of your card being used in two countries simultaneously.

> **Engineer:** Compare the timezone implied by id_14 against the timezone implied by addr1 (billing zip). A mismatch of more than 3 hours is a strong fraud indicator.

---

### `id_15` — Cookie/Session Match
Values: `Found` / `New` / `Unknown`

**What it means:** Vesta sets a tracking cookie in the customer's browser.
- `Found` = this browser has Vesta's cookie from a previous session → returning customer, lower risk
- `New` = first time this browser has visited → no history, unknown risk
- `Unknown` = cookie was blocked or browser is in private/incognito mode → **suspicious!** Fraudsters use incognito or cookie-blocking to avoid being tracked

> **Account takeover tells:** If id_15 = `Found` (returning customer cookie) BUT id_28 = `New` (browser fingerprint never seen before), that means: someone has the victim's cookie (session hijacking) but is using a different device/browser. This is a major account takeover signal.

---

### `id_16` — Browser Fingerprint Match
Values: `Found` / `NotFound`

The browser fingerprint is a unique signature derived from: browser version + OS + installed fonts + screen resolution + timezone + language. 
- `Found` = this exact browser configuration was used before
- `NotFound` = brand new device/browser configuration

Fraudsters using virtual machines or fresh browser instances will always be `NotFound`.

---

### `id_17` through `id_20` — Behavioral Scores
Numeric scores. These appear to be risk/behavioral scores at different aggregation levels (session-level, device-level, network-level, account-level). Higher absolute values indicate more anomalous behavior.

---

### `id_21` through `id_27` — Network/Connection Features
**96%+ missing** — these are only captured when Vesta's full fraud SDK is active on the merchant's site.

When present, these likely capture:
- Number of IPs associated with this device
- VPN/proxy detection score
- Tor exit node detection
- Network anonymization indicators

> **Why this matters:** A fraudster will almost always use a VPN, Tor, or proxy to hide their real location. When id_21–id_27 are populated and show anomalous values, it's a very strong fraud signal. The 96% missingness is unfortunate — but when these values ARE present, treat them as gold.

---

### `id_28` — Browser Fingerprint History
Values: `Found` / `New`

Similar to id_16 but with slightly different scope. This may check a longer historical window:
- `Found` = this browser fingerprint is in Vesta's long-term database
- `New` = never seen before on Vesta's entire network (not just this merchant)

---

### `id_29` — Something Looked Up / Verified
Values: `Found` / `NotFound` / `New`

Likely a network-level lookup — whether the device's characteristics were found in Vesta's cross-merchant database.

---

### `id_30` — Operating System (Raw String)
Examples: `Android 7.0`, `iOS 11.1.2`, `Windows`, `Mac OS X 10_13_6`

**What to engineer:**
- OS family: Windows / iOS / Android / Mac / Linux
- OS version (older OS = higher risk; fraudsters often use old unpatched systems)
- Flag: does the OS match the DeviceType? (`iOS` should be `mobile`, `Windows` should be `desktop` — mismatch = possible spoofing)

---

### `id_31` — Browser (Raw String)
Examples: `chrome 67.0 for android`, `mobile safari 11.0`, `samsung browser 6.2`

**What to engineer:**
- Browser family: Chrome / Safari / Firefox / IE / Samsung
- Browser version (old browser = suspicious)
- Flag: `Trident/7.0` = Internet Explorer 11 — this is an old, dying browser; transactions from IE11 have higher fraud rates in practice
- Mobile vs. desktop browser (should match DeviceType)

**The fraud signal:** A fraudster using automation tools (like Selenium WebDriver or PhantomJS) will show a headless browser user-agent string — this almost never appears in legitimate traffic.

---

### `id_32` — Screen Resolution Width
Numeric value (e.g., 32 appears in the data, but actual values like 1280, 1920 are more typical).

Virtual machines used for automated fraud often have non-standard screen resolutions. A resolution of 800×600 in 2019 is almost never a real user.

---

### `id_33` — Screen Resolution String
Format: `"1280x720"`, `"2220x1080"`, `"1334x750"`, etc.

**Engineer:** Extract width × height, compute aspect ratio, flag unusual resolutions. A resolution that doesn't correspond to any known device model is a spoofing indicator.

---

### `id_34` — Match Status Code
Format: `"match_status:2"`, `"match_status:1"`, etc.

This is an internal Vesta match scoring result — likely a composite of multiple identity checks. Higher numbers may indicate more verification steps passed or failed. Treat the numeric part as a categorical feature.

---

### `id_35` through `id_38` — Binary Flags
Values: `T` (True) / `F` (False)

These are additional binary verification results. Based on the data patterns:
- **id_35**: Possibly whether the browser supports a specific security feature (e.g., JavaScript enabled, cookies enabled)
- **id_36**: Could be a headless browser detection flag (F = headless browser detected → fraud risk!)
- **id_37**: Possibly whether the device passed a device fingerprinting challenge
- **id_38**: Possibly whether the session passed a behavioral biometrics check

> **The pattern to watch:** Legitimate users tend to have id_35=T, id_37=T, id_38=T. Fraudster bots often fail behavioral checks — id_36=F, id_38=F.

---

### `DeviceType`
Values: `desktop` / `mobile`

**Desktop** = higher-value transactions, more deliberate purchases, but also easier to automate attacks (bots are usually desktop).  
**Mobile** = increasingly common for fraud (mobile malware, SIM-swap attacks), but harder to automate perfectly.

Fraud rates differ between device types — always include this as a feature.

---

### `DeviceInfo`
Raw string: `"SAMSUNG SM-G892A Build/NRD90M"`, `"iOS Device"`, `"Windows"`, `"Trident/7.0"`

**This is the single richest raw text field in the dataset.** It contains:
- Device manufacturer (Samsung, Apple, Huawei, LG)
- Device model number
- Build/firmware version
- For desktop: browser engine (Trident = IE, rv:57 = Firefox 57)

**What to engineer:**
1. Device brand (Samsung / Apple / Huawei / Windows-PC / Mac)
2. Is it a known high-fraud device model?
3. Is it a browser engine string pretending to be a device? (Trident/7.0, rv:11.0 — these are IE/Firefox user agents, not real device names)
4. Does the manufacturer match the OS? (Samsung brand + iOS = spoofed DeviceInfo)

---

## PART 5 — How to Read a Fraudulent Transaction

Here is an example of what a **high-fraud transaction looks like** using this domain understanding:

```
TransactionAmt  = $1,897.00      ← Large amount, unusual
ProductCD       = C              ← Gift card / cash equivalent
card1           = [same as 40 other transactions in the last 2 hours]  ← Card testing
card4           = visa
card6           = credit         ← Credit card (more time before victim notices)
addr1           = 10245          ← Billing zip
dist1           = 1,847 miles    ← Shipping address 1,847 miles away  🚨
P_emaildomain   = anonymous.com  🚨
R_emaildomain   = protonmail.com 🚨  (different from purchaser)
C1              = 12             ← 12 different billing addresses on this card
C4              = 8              ← 8 failed transactions before this one (card testing)
D1              = 0.002          ← Last transaction was 3 minutes ago  🚨
M1              = F              ← Name doesn't match  🚨
M2              = F              ← CVV mismatch  🚨
M3              = F              ← Address mismatch  🚨
id_14           = 480            ← Browser in UTC+8 (Asia), card billing in US  🚨
id_15           = New            ← First-ever visit to Vesta
id_28           = New            ← Device never seen before
id_30           = Windows        ← Desktop (bot-friendly)
id_31           = chrome 67.0    ← Slightly old browser (automated)
id_36           = F              ← Failed a behavioral check  🚨
DeviceType      = desktop
isFraud         = 1              ← CONFIRMED FRAUD
```

---

## PART 6 — The Evaluation Lens: What Must You Watch?

| What to Watch | Why It Matters | The Wrong Approach |
|---|---|---|
| **Class imbalance** | 96.5% legitimate means a dumb model wins on accuracy | Never use accuracy as your metric |
| **Time-based split** | Fraud patterns evolve; past cannot predict future if you leak | Never use random train/val splits |
| **V-feature groups** | Highly correlated within groups; redundancy kills efficiency | Don't treat all 339 as independent |
| **Missing values** | NaN is not random — it's information | Don't impute with global mean blindly |
| **Card-level aggregation** | Card velocity is the #1 fraud signal | Don't treat each row as independent |
| **The identity join** | 76% of transactions have NO identity record | Don't drop rows without identity; flag them |
| **id_14 (timezone)** | The most underrated feature in the entire dataset | Don't ignore timezone mismatch |
| **M-flag combinations** | Multiple F flags together = near-certain fraud | Don't treat M-flags as individual features only; combine them |
| **anonymous.com** | A deliberate fraud signal hidden in a domain name | Don't bucket all email domains the same way |

---

*"The dataset doesn't just contain numbers. It contains the digital fingerprints of real fraudsters and real victims. Every row is a moment where someone, somewhere, was either protected or robbed. Your model's job is to tell the difference."*

---
*Deep domain analysis prepared for credit-card-fraud-detection-system project.*
