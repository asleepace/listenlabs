import type { Keys, GameState, Person } from "..";

interface Constraint {
  attribute: string;
  minCount: number;
}

interface GameData {
  gameId: string;
  constraints: Constraint[];
  attributeStatistics: {
    relativeFrequencies: Record<string, number>;
    correlations: Record<string, Record<string, number>>;
  };
}

interface GameCounter {
  state: GameState;
  admit(status: GameState['status']): boolean;
}

export class NightclubGameCounter implements GameCounter {
  private gameData: GameData;
  private attributeCounts: Record<string, number> = {};
  private maxCapacity = 1000;
  private totalPeople = 10000;
  state: GameState;

  constructor(initialData: GameState) {
    this.gameData = initialData.game;
    this.state = initialData;
    
    // Initialize attribute counts
    this.gameData.constraints.forEach(constraint => {
      this.attributeCounts[constraint.attribute] = 0;
    });
  }

  admit(status: GameState['status']): boolean {
    // Update state status
    this.state.status = status;
    const { nextPerson } = status
    
    if (!nextPerson) {
      return false;
    }

    const spotsLeft = this.maxCapacity - this.admittedCount;
    const peopleInLineLeft = this.totalPeople - this.admittedCount - this.rejectedCount;
    
    // If we're at capacity, reject
    if (spotsLeft <= 0) {
      return false;
    }

    const personAttributes = nextPerson.attributes;
    
    // Calculate admission score
    const score = this.calculateAdmissionScore(
      personAttributes,
      spotsLeft,
      peopleInLineLeft
    );
    
    // Dynamic threshold based on how many spots are left
    const progressRatio = this.admittedCount / this.maxCapacity;
    const baseThreshold = 0.3;
    const threshold = baseThreshold * (1 - progressRatio * 0.5); // More lenient early, stricter later
    
    const shouldAdmit = score > threshold;
    
    if (shouldAdmit) {
      // Update counts for admitted person
      Object.entries(personAttributes).forEach(([attr, hasAttr]) => {
        if (hasAttr && this.attributeCounts[attr] !== undefined) {
          this.attributeCounts[attr]++;
        }
      });
    }
    
    return shouldAdmit;
  }

  get admittedCount() {
    if (this.state.status.status !== 'running') throw new Error('not_running')
    return this.state.status.admittedCount
  }

  get rejectedCount() {
    if (this.state.status.status !== 'running') throw new Error('not_running')
    return this.state.status.rejectedCount
  }

  private calculateAdmissionScore(
    attributes: Record<string, boolean>,
    spotsLeft: number,
    peopleInLineLeft: number
  ): number {
    let score = 0;
    const frequencies = this.gameData.attributeStatistics.relativeFrequencies;
    
    // Check if all quotas are already met
    const allQuotasMet = this.gameData.constraints.every(constraint => 
      this.attributeCounts[constraint.attribute]! >= constraint.minCount
    );
    
    // If all quotas are met, admit everyone
    if (allQuotasMet || this.state.status.status !== 'running') {
      return 10.0; // High score to guarantee admission
    }

    const { admittedCount, rejectedCount } = this.state.status
    const totalProcessed = admittedCount + rejectedCount;
    
    // Calculate score for each attribute the person has
    this.gameData.constraints.forEach(constraint => {
      const attr = constraint.attribute;
      
      if (!attributes[attr]) return;
      
      const currentCount = this.attributeCounts[attr]!;
      const needed = constraint.minCount - currentCount;
      
      if (needed <= 0) return; // Quota already met
      
      const frequency = frequencies[attr] || 0.5;
      const expectedRemaining = peopleInLineLeft * frequency;
      
      // Expected progress: where we should be at this point in the line
      // We want to fill quotas by person 5000 (halfway through the line)
      const targetProgress = Math.min(totalProcessed / 5000, 1.0);
      const actualProgress = currentCount / constraint.minCount;
      const progressGap = targetProgress - actualProgress;
      
      // Base urgency: how behind schedule are we?
      const urgency = progressGap > 0 ? progressGap * 2 : 0;
      
      // Scarcity factor: how rare is this attribute?
      const scarcityFactor = 1 / Math.max(frequency, 0.01);
      
      // Risk factor: can we afford to wait?
      const riskFactor = needed / Math.max(expectedRemaining, 1);
      
      // Component score combines all factors
      const componentScore = (urgency + riskFactor) * Math.log(scarcityFactor + 1);
      
      // Add correlation bonus for multiple needed attributes
      let correlationBonus = 0;
      this.gameData.constraints.forEach(otherConstraint => {
        if (otherConstraint.attribute === attr) return;
        
        const otherAttr = otherConstraint.attribute;
        const otherNeeded = otherConstraint.minCount - this.attributeCounts[otherAttr]!;
        
        if (attributes[otherAttr] && otherNeeded > 0) {
          const correlation = this.gameData.attributeStatistics.correlations[attr]?.[otherAttr] || 0;
          // Positive correlation is good when both are needed
          correlationBonus += correlation * 0.2;
        }
      });
      
      score += componentScore * (1 + correlationBonus);
    });
    
    // Bonus for having multiple useful attributes
    const usefulAttributes = Object.entries(attributes).filter(([attr, has]) => {
      if (!has) return false;
      const constraint = this.gameData.constraints.find(c => c.attribute === attr);
      if (!constraint) return false;
      return this.attributeCounts[attr]! < constraint.minCount;
    }).length;
    
    // Give bonus for multiple attributes but don't over-penalize
    if (usefulAttributes > 1) {
      score *= (1 + 0.2 * (usefulAttributes - 1));
    }
    
    return score;
  }
  
  // Helper method to check current progress
  getProgress(): {
    quotasMet: Record<string, boolean>;
    quotaProgress: Record<string, number>;
    admissionRate: number;
  } {
    const quotasMet: Record<string, boolean> = {};
    const quotaProgress: Record<string, number> = {};
    
    this.gameData.constraints.forEach(constraint => {
      const attr = constraint.attribute;
      quotasMet[attr] = this.attributeCounts[attr]! >= constraint.minCount;
      quotaProgress[attr] = this.attributeCounts[attr]! / constraint.minCount;
    });
    
    return {
      quotasMet,
      quotaProgress,
      admissionRate: this.admittedCount / (this.admittedCount + this.rejectedCount)
    };
  }
}