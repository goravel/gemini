package facades

import (
	"fmt"

	contractsai "github.com/goravel/framework/contracts/ai"

	"github.com/goravel/gemini"
)

func Gemini(provider string) (contractsai.Provider, error) {
	if gemini.App == nil {
		return nil, fmt.Errorf("please register gemini service provider")
	}

	instance, err := gemini.App.MakeWith(gemini.Binding, map[string]any{
		"provider": provider,
	})
	if err != nil {
		return nil, err
	}

	providerInstance, ok := instance.(contractsai.Provider)
	if !ok {
		return nil, fmt.Errorf("resolved gemini binding is %T, not ai.Provider", instance)
	}

	return providerInstance, nil
}
