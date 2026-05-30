package main

import (
	"os"

	"github.com/goravel/framework/packages"
	"github.com/goravel/framework/packages/match"
	"github.com/goravel/framework/packages/modify"
	"github.com/goravel/framework/support/path"
)

func main() {
	setup := packages.Setup(os.Args)
	aiConfigPath := path.Config("ai.go")
	moduleImport := setup.Paths().Module().Import()
	serviceProvider := "&gemini.ServiceProvider{}"
	aiProviderContract := "github.com/goravel/framework/contracts/ai"
	geminiFacadesImport := moduleImport + "/facades"
	provider := `map[string]any{
		"key": config.Env("GEMINI_API_KEY", ""),
		"models": map[string]any{
			"text": map[string]any{
				"default": "",
			},
			"audio": map[string]any{
				"default": "",
			},
			"transcription": map[string]any{
				"default": "",
			},
			"image": map[string]any{
				"default": "",
			},
		},
		"url": config.Env("GEMINI_API_URL", ""),
		"via": func() (ai.Provider, error) {
			return geminifacades.Gemini("gemini")
		},
	}`
	aiProvidersConfig := match.Config("ai.providers")

	setup.Install(
		modify.RegisterProvider(moduleImport, serviceProvider),

		modify.GoFile(aiConfigPath).Find(match.Imports()).Modify(
			modify.AddImport(aiProviderContract),
			modify.AddImport(geminiFacadesImport, "geminifacades"),
		).Find(aiProvidersConfig).Modify(modify.AddConfig("gemini", provider)),
	).Uninstall(
		modify.WhenFileExists(aiConfigPath, modify.GoFile(aiConfigPath).
			Find(aiProvidersConfig).Modify(modify.RemoveConfig("gemini")).
			Find(match.Imports()).Modify(
			modify.RemoveImport(aiProviderContract),
			modify.RemoveImport(geminiFacadesImport, "geminifacades"),
		)),

		modify.UnregisterProvider(moduleImport, serviceProvider),
	).Execute()
}
