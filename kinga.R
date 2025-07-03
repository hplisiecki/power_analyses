# ============================================
# 0. Install & load libraries
# ============================================
# install.packages(c("lme4","simr","emmeans","ggplot2"))
library(lme4)       # for lmer()
library(simr)       # for extend(), powerSim()
library(emmeans)    # for emmeans(), contrast()
library(ggplot2)    # for plotting
library(tidyr)    # for pivot_longer()

# ============================================
# 1. Simulate a tiny “pilot” (n0 = 30)
#    with guessed σ_u = 1.5 and σ_ε = 2
# ============================================
set.seed(2025)
n0 <- 30
pilot <- expand.grid(
  participant = factor(1:n0),
  content     = factor(c("neutral","stereotypical")),
  author      = factor(c("female","male")),
  partner     = factor(c("neutral","female","male"))
)
u   <- rnorm(n0, 0, 1.5)
eps <- rnorm(nrow(pilot), 0, 2)
pilot$amount <- 5 + u[pilot$participant] + eps

# ============================================
# 2. Fit “null” full model and impose small effects
# ============================================
m0 <- lmer(
  amount ~ content * author * partner + (1 | participant),
  data = pilot
)

# grab & overwrite fixed effects:
fe <- fixef(m0)
fe["contentstereotypical"]                          <- -0.2   # H1
fe["contentstereotypical:authormale:partnermale"]   <- -0.2   # H2
fe["contentstereotypical:authormale:partnerfemale"] <- +0.2   # H4
# zero any others so they don't bleed into our tests
fe[c(
  "contentstereotypical:authormale",
  "contentstereotypical:partnerfemale",
  "contentstereotypical:partnerneutral",
  "contentstereotypical:authormale:partnerneutral"
)] <- 0
m0 <- setFixef(m0, fe)

# pre‐define the contrast vector for H3 (Δ_mm − Δ_ff)
# emmeans order: ff, fm, fn, mf, mm, mn  (× 2 contents each)
cvec3 <- c(
   1, -1,  # female×female
   0,  0,  # female×male
   0,  0,  # female×neutral
   0,  0,  # male×female
  -1,  1,  # male×male
   0,  0   # male×neutral
)

# ============================================
# 3. Define grid of sample sizes & empty results
# ============================================
Ns   <- seq(200, 2000, by = 100)
nsim <- 200
res  <- expand.grid(N = Ns, H = paste0("H",1:4), POWER = NA_real_)
# reshape into wide later

# ============================================
# 4. Loop over Ns and hypotheses
# ============================================
row <- 1
for(N in Ns) {
  cat("Simulating N =", N, "...\n")
  mN <- extend(m0, along = "participant", n = N)
  
  # H1: main effect contentstereotypical
  p1 <- powerSim(mN, fixed("contentstereotypical"), nsim = nsim)$power
  
  # H2: 3-way content:authormale:partnermale
  p2 <- powerSim(
    mN,
    fixed("contentstereotypical:authormale:partnermale"),
    nsim = nsim
  )$power
  
  # H3: Δ_mm − Δ_ff
  emm3 <- emmeans(mN, ~ content | author * partner)
  p3   <- powerSim(
    mN,
    contrast(emm3, method = list(H3 = cvec3)),
    nsim = nsim
  )$power
  
  # H4: simple effect at male author→female partner
  emm4 <- emmeans(
    mN,
    ~ content | author * partner,
    at = list(author = "male", partner = "female")
  )
  p4   <- powerSim(
    mN,
    contrast(emm4, "trt.vs.ctrl", ref = "neutral"),
    nsim = nsim
  )$power
  
  # store
  res[row:(row+3), "POWER"] <- c(p1, p2, p3, p4)
  row <- row + 4
}

# ============================================
# 5. Tidy & plot
# ============================================
library(dplyr)
df <- res %>%
  pivot_wider(names_from = H, values_from = POWER)

print(df)

# long format for ggplot
df_long <- df %>% pivot_longer(-N, names_to = "Hypothesis", values_to = "Power")

ggplot(df_long, aes(x = N, y = Power, color = Hypothesis)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_y_continuous(labels = scales::percent_format(accuracy=1)) +
  labs(
    x = "Number of Participants",
    y = "Estimated Power",
    title = "Power Curves for H1–H4 in Study 1",
    color = "Hypothesis"
  ) +
  theme_minimal()
